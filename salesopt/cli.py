"""명령줄 진입점:  python -m salesopt <명령>

  demo        가짜 데이터로 학습 → 굿즈 종류별 추천 가격·안전 수량 (네트워크·DB 불필요)
  backtest    가짜 데이터로 '정말 안전한가' 검증 (데이터 양·모델·보정 방식별)
  template    실제 판매 기록 CSV 양식 만들기
  ingest-idol IdolSentiments 결과를 DB 에 날짜별로 저장 (매일 실행)
  recommend   실제 데이터로 특정 그룹의 굿즈 추천 (판매 기록 CSV + DB 특징)
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from salesopt.conformal import LEVEL_GRID, LevelConformal, min_records
from salesopt.models import fit_predictor
from salesopt.optimize import CostSpec, recommend
from salesopt.simulate import GOODS_TYPES, NUMERIC_FEATURES, Simulator, default_cost

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("salesopt")

SHOW_COLS = ["price", "safe_qty", "sellout_prob", "expected_profit", "expected_leftover", "model_agreement",
             "demand_p10", "demand_p50", "newsvendor_qty", "newsvendor_sellout_prob"]


def _model_kw(a) -> dict:
    return {"min_elasticity": a.min_elasticity} if a.model == "lognormal" and a.min_elasticity else {}


def _won(v) -> str:
    return f"{v:,.0f}원"


def print_recommendation(title: str, rec, show_table: bool = True) -> None:
    print(f"\n=== {title} ===")
    if rec.best:
        b = rec.best
        print(f"  추천: 가격 {_won(b['price'])} × {b['safe_qty']:,}개   "
              f"완판 확률 {b['sellout_prob']:.0%} (목표 ≥ {1 - rec.alpha:.0%})   기대이익 {_won(b['expected_profit'])}")
        print(f"        참고) 논문 원래 기준(기대이익 최대)이면 {b['newsvendor_qty']:,}개 → 완판 확률 "
              f"{b['newsvendor_sellout_prob']:.0%}")
    else:
        print("  추천 없음: 완판 확률을 지키면서 이익이 나는 가격이 없습니다 → 만들지 않는 것을 권장")
    for n in rec.notes:
        print(f"  ※ {n}")
    if show_table and not rec.table.empty:
        t = rec.table[SHOW_COLS].copy()
        t["sellout_prob"] = t.sellout_prob.map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
        t["newsvendor_sellout_prob"] = t.newsvendor_sellout_prob.map(lambda v: f"{v:.0%}")
        t["model_agreement"] = t.model_agreement.map(lambda v: f"{v:.0%}")
        for c in ("expected_profit", "expected_leftover", "demand_p10", "demand_p50"):
            t[c] = t[c].round(0).astype(int)
        print(t.to_string(index=False))


# ---------------------------------------------------------------------------- demo / backtest
def cmd_demo(a) -> None:
    sim = Simulator(seed=a.seed)
    train = sim.records(a.n)
    print(f"가짜 판매 기록 {len(train)}건 학습 (완판 비율 {train.sold_out.mean():.0%}) · 모델={a.model}")
    pred = fit_predictor(train, NUMERIC_FEATURES, kind=a.model, n_members=a.members, seed=a.seed, **_model_kw(a))
    base = sim.contexts(1).iloc[0].to_dict()
    base.update(favor=72, interest=85, controversy=0.15, log_mentions_per_day=6.5, favor_change_7d=2.0,
                member_hhi=0.3, hot_index=70, buzz_z=1.0, sold_out_ratio=0.4, group_type_buzz_7d=20)
    from salesopt.backtest import price_grid
    for gt in (a.types.split(",") if a.types else GOODS_TYPES):
        ctx = pd.DataFrame([base | {"goods_type": gt}])
        rec = recommend(pred, ctx, price_grid(gt), default_cost(gt), alpha=a.alpha, m=a.samples, seed=a.seed)
        print_recommendation(f"[가짜 그룹] {gt}", rec, show_table=not a.brief)
        if rec.best:
            p, q = rec.best["price"], rec.best["safe_qty"]
            print(f"  (정답 확인: 이 가격·수량의 참 완판 확률 {sim.true_sellout_prob(ctx, p, q)[0]:.0%})")


def cmd_backtest(a) -> None:
    from salesopt.backtest import run_backtest
    keys = ["safe_sellout_mean", "safe_violation_rate", "safe_profit_ratio", "safe_make_rate",
            "nv_sellout_mean", "nv_profit_ratio"]
    rows = []
    for n in [int(x) for x in a.n_train.split(",")]:
        rs = [run_backtest(n_train=n, n_calib=a.n_calib, n_test=a.n_test, alpha=a.alpha, kind=a.model,
                           n_members=a.members, confounding=a.confounding, seed=s, censored=a.censored,
                           model_kw=_model_kw(a))
              for s in range(a.seeds)]
        row = {"n_train": n, "n_calib": a.n_calib} | {k: float(np.mean([r[k] for r in rs])) for k in keys}
        rows.append(row)
        print(pd.DataFrame([row]).round(3).to_string(index=False, header=len(rows) == 1), flush=True)
    print(f"\n목표 완판 확률 {1 - a.alpha:.0%}. safe = 이 프로그램, nv = 논문 원래 기준(기대이익 최대).")
    print("profit_ratio = 참 분포를 아는 '안전 기준 최선' 대비 기대이익 비율 (nv 는 안전 기준이 없어 1을 넘을 수 있음)")


# ---------------------------------------------------------------------------- 실제 데이터
def cmd_template(a) -> None:
    from salesopt.schema import write_template
    print(f"저장: {write_template(Path(a.out))}")


def cmd_ingest_idol(a) -> None:
    from salesopt.sources.idol import DEFAULT_SOURCE, ingest
    from salesopt.storage import init_db, session_scope
    init_db()
    with session_scope() as s:
        print(ingest(s, a.source or DEFAULT_SOURCE))


def load_costs(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _cost_and_grid(costs: dict, goods_type: str, support) -> tuple[CostSpec, np.ndarray, list[str]]:
    c = (costs.get("goods") or {}).get(goods_type) or {}
    d = costs.get("default") or {}
    get = lambda k, v=None: c.get(k, d.get(k, v))                                  # noqa: E731
    if get("unit_cost") is None:
        raise SystemExit(f"costs 파일에 '{goods_type}' 의 unit_cost(개당 원가)가 없습니다")
    cost = CostSpec(unit_cost=float(get("unit_cost")), salvage=float(get("salvage", 0)),
                    fixed_cost=float(get("fixed_cost", 0)), moq=int(get("moq", 1)), max_qty=get("max_qty"))
    notes = []
    lo, hi = get("price_min"), get("price_max")
    if support:
        if lo is None or hi is None:
            lo, hi = lo or support[0], hi or support[1]
        if lo < support[0] or hi > support[1]:
            notes.append(f"가격 범위 {lo:,.0f}~{hi:,.0f}원 중 학습 데이터 범위({support[0]:,.0f}~{support[1]:,.0f}원) "
                         "밖은 외삽이라 덜 믿을 만합니다")
    if lo is None or hi is None:
        raise SystemExit(f"'{goods_type}' 가격 범위(price_min, price_max)를 costs 파일에 넣어주세요")
    step = float(get("price_step", 500))
    grid = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    return cost, grid, notes


def cmd_recommend(a) -> None:
    from salesopt.features import (build_context, build_training_frame, feature_coverage, load_group_map,
                                   resolve_group, select_candidates)
    from salesopt.schema import load_sales_csv
    from salesopt.storage import init_db, session_scope

    init_db()
    gmap = load_group_map()
    costs = load_costs(Path(a.costs))
    before = date.fromisoformat(a.as_of) + timedelta(days=1) if a.as_of else date.today() + timedelta(days=1)
    with session_scope() as s:
        gkey = resolve_group(a.group, gmap, s)
        group = gmap[gkey]
        sales = load_sales_csv(Path(a.sales), lambda g: resolve_group(g, gmap, s))
        data = build_training_frame(s, sales, gmap)
        cov = feature_coverage(data)
        numeric = [f for f in NUMERIC_FEATURES if cov[f] >= a.min_coverage]
        dropped = [f for f in NUMERIC_FEATURES if f not in numeric]
        print(f"판매 기록 {len(data)}건 (완판 {int(data.sold_out.sum())}건) · 그룹 {data.group_key.nunique()}개 · "
              f"굿즈 종류 {data.goods_type.nunique()}개")
        if dropped:
            print(f"  값이 부족해 뺀 특징: {', '.join(dropped)}")

        n_cal = int(len(data) * a.calib_frac)
        conf = None
        if n_cal >= min_records(a.alpha):
            perm = np.random.default_rng(a.seed).permutation(len(data))
            cal, train = data.iloc[perm[:n_cal]], data.iloc[perm[n_cal:]]
        else:
            cal, train = None, data
            print(f"  보정용 기록이 부족해 conformal 보정을 하지 않습니다 (α={a.alpha} 이면 보정용 최소 "
                  f"{min_records(a.alpha)}건, 현재 {n_cal}건)")
        pred = fit_predictor(train, numeric, kind=a.model, n_members=a.members, seed=a.seed, **_model_kw(a))
        if cal is not None:
            L = pred.lower_bounds(cal, cal.price.to_numpy(), LEVEL_GRID, seed=a.seed)
            conf = LevelConformal(a.alpha, censored=a.censored).fit(L, cal.quantity_sold, cal.sold_out)
            print(f"  conformal 보정: 기록 {conf.n}건 (완판 {conf.n_censored}건) → 분위수 수준 "
                  f"{a.alpha:.3f} → {conf.level:.3f}")

        types = a.types.split(",") if a.types else select_candidates(s, group, before, a.top)
        if not types:
            raise SystemExit("후보 굿즈 종류가 없습니다 (--types 로 지정하거나 Goods 점수 계산을 확인하세요)")
        known = set(pred.encoder.categories_)
        summary = []
        for gt in types:
            if gt not in known:
                print(f"\n=== {gt} === 판매 기록에 없는 굿즈 종류라 추천할 수 없습니다")
                continue
            ctx = build_context(s, group, gt, before)
            cost, grid, notes = _cost_and_grid(costs, gt, pred.price_range(gt))
            rec = recommend(pred, ctx, grid, cost, alpha=a.alpha, m=a.samples, conformal=conf, seed=a.seed)
            rec.notes = notes + rec.notes
            print_recommendation(f"{group.key} · {gt}", rec, show_table=not a.brief)
            if rec.best:
                summary.append({"goods_type": gt} | {k: rec.best[k] for k in
                                                     ("price", "safe_qty", "sellout_prob", "expected_profit")})
    if summary:
        out = pd.DataFrame(summary).sort_values("expected_profit", ascending=False)
        print("\n=== 굿즈별 요약 (기대이익 순) ===")
        print(out.to_string(index=False))
        path = ROOT / "outputs" / f"{gkey}_{before - timedelta(days=1)}.csv"
        path.parent.mkdir(exist_ok=True)
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"저장: {path}")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="salesopt", description="아이돌 굿즈 안전 완판 가격·수량 추천")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, members=10):
        q.add_argument("--alpha", type=float, default=0.1, help="허용 미완판 확률 (0.1 → 완판 확률 90%%)")
        q.add_argument("--model", choices=["lognormal", "cvae"], default="lognormal")
        q.add_argument("--members", type=int, default=members, help="앙상블 크기")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--min-elasticity", type=float, default=0.0,
                       help="가격 탄력성 하한 (lognormal). 과거 가격이 수요를 보고 정해졌다면 0.5~1.0 권장")

    d = sub.add_parser("demo"); common(d)
    d.add_argument("--n", type=int, default=300, help="가짜 판매 기록 수")
    d.add_argument("--types", default="")
    d.add_argument("--samples", type=int, default=4000)
    d.add_argument("--brief", action="store_true")

    b = sub.add_parser("backtest"); common(b)
    b.add_argument("--n-train", default="50,300,2000")
    b.add_argument("--n-calib", type=int, default=0)
    b.add_argument("--n-test", type=int, default=80)
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--censored", choices=["strict", "impute"], default="strict")
    b.add_argument("--confounding", action="store_true", help="가격이 수요에 맞춰 정해진 과거 데이터")

    t = sub.add_parser("template"); t.add_argument("--out", default="sales_template.csv")

    i = sub.add_parser("ingest-idol"); i.add_argument("--source", default="", help="URL 또는 로컬 폴더")

    r = sub.add_parser("recommend"); common(r)
    r.add_argument("--group", required=True)
    r.add_argument("--sales", required=True, help="실제 판매 기록 CSV (python -m salesopt template 참고)")
    r.add_argument("--costs", default=str(ROOT / "config" / "costs.yaml"))
    r.add_argument("--types", default="", help="굿즈 종류 (쉼표). 비우면 Goods 점수로 후보 선정")
    r.add_argument("--top", type=int, default=5)
    r.add_argument("--as-of", default="", help="기준일 (기본 오늘)")
    r.add_argument("--calib-frac", type=float, default=0.3, help="conformal 보정에 쓸 기록 비율")
    r.add_argument("--censored", choices=["strict", "impute"], default="strict")
    r.add_argument("--min-coverage", type=float, default=0.5, help="값이 이 비율 미만인 특징은 제외")
    r.add_argument("--samples", type=int, default=4000)
    r.add_argument("--brief", action="store_true")

    a = p.parse_args(argv)
    {"demo": cmd_demo, "backtest": cmd_backtest, "template": cmd_template, "ingest-idol": cmd_ingest_idol,
     "recommend": cmd_recommend}[a.cmd](a)


if __name__ == "__main__":
    main()
