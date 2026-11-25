"""웹(web/)이 쓸 모델·특징을 Supabase 에 올린다 + 웹과 똑같이 계산하는 '결정적' 추천 함수.

웹은 Python 모델을 돌리지 않는다. 대신
  so_model   : lognormal 앙상블의 파라미터 (멤버별 계수, 인코더 평균·표준편차, 보정된 분위수 수준 …)
  so_context : (그룹, 굿즈 종류)별 현재 특징 (features.build_context 결과)
를 읽어서 TypeScript(web/lib/optimize.ts)로 추천을 즉시 계산한다.

결정적 계산 (샘플링 대신 정확한 식/수치적분 → 새로고침해도 결과가 같다)
  멤버 b 의 수요: log(1+D) ~ N(μ_b, σ_b²)
  안전 수량      : 멤버별 a 분위수 expm1(μ_b + σ_b Φ⁻¹(a)) 의 conservative 백분위 (optimize.recommend 와 같은 규칙)
  완판 확률      : 멤버 평균 P_b(D ≥ q) = mean_b Φc((log(1+q) − μ_b)/σ_b)
  기대 판매량    : E[min(q, D)] 를 z ∈ [−8, 8] 격자 적분
recommend() (몬테카를로) 결과와 거의 같아야 한다 → tests/test_publish.py 가 확인.
"""
from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from salesopt.models import fit_predictor
from salesopt.models.lognormal import LogNormalDemand
from salesopt.optimize import CostSpec, critical_ratio
from salesopt.simulate import GOODS_TYPES, NUMERIC_FEATURES, Simulator

MIN_REAL_RECORDS = 20           # 실제 판매 기록이 이보다 적으면 실제 모델을 만들지 않는다
Z_GRID = np.linspace(-8, 8, 801)
Z_W = norm.pdf(Z_GRID)
Z_W = Z_W / Z_W.sum()


# --------------------------------------------------------------------------- 모델 → JSON
def export_model(pred, kind: str, meta: dict | None = None, conformal_level: float | None = None,
                 conservative: float = 0.1) -> dict:
    enc = pred.encoder
    members = []
    for m in pred.members:
        if not isinstance(m, LogNormalDemand):
            raise ValueError("웹에는 lognormal 모델만 올릴 수 있습니다")
        b0, b, th, a0, a = m._unpack(m.w_)
        members.append({"b0": float(b0), "b": b.tolist(), "a0": float(a0), "a": a.tolist(),
                        "elasticity": m.elasticity().tolist(), "lp_center": m.lp_center_.tolist()})
    return {
        "version": 1, "kind": kind, "numeric": list(enc.numeric), "mean": enc.mean_.tolist(),
        "std": enc.std_.tolist(), "categories": list(enc.categories_), "members": members,
        "conservative": conservative, "conformal_level": conformal_level,
        "price_support": {k: list(v) for k, v in pred.price_support.items()},
        "meta": meta or {},
    }


def _encode(model: dict, features: dict, goods_type: str) -> tuple[np.ndarray, int]:
    vals = np.array([np.nan if features.get(f) is None else float(features[f]) for f in model["numeric"]])
    num = np.nan_to_num((vals - np.array(model["mean"])) / np.array(model["std"]), nan=0.0)
    cats = model["categories"]
    if goods_type not in cats:
        raise KeyError(goods_type)
    g = cats.index(goods_type)
    onehot = np.zeros(len(cats) - 1)
    if g > 0:
        onehot[g - 1] = 1.0
    return np.concatenate([num, onehot]), g


def member_params(model: dict, features: dict, goods_type: str, price: float) -> tuple[np.ndarray, np.ndarray]:
    x, g = _encode(model, features, goods_type)
    mu, sg = [], []
    for m in model["members"]:
        mu.append(m["b0"] + x @ np.array(m["b"]) - m["elasticity"][g] * (math.log(price) - m["lp_center"][g]))
        sg.append(math.exp(min(max(m["a0"] + x @ np.array(m["a"]), -4.0), 3.0)))
    return np.array(mu), np.array(sg)


def _mixture_quantile(mu, sg, level: float) -> float:
    """혼합분포 (멤버 평균) 의 수요 level 분위수."""
    lo, hi = float((mu - 9 * sg).min()), float((mu + 9 * sg).max())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if norm.cdf((mid - mu) / sg).mean() < level:
            lo = mid
        else:
            hi = mid
    return max(math.floor(math.expm1(hi)), 0)


def _expected_sold(mu, sg, q: float) -> float:
    d = np.maximum(np.floor(np.expm1(mu[:, None] + sg[:, None] * Z_GRID[None, :])), 0)
    return float((np.minimum(q, d) * Z_W[None, :]).sum(axis=1).mean())


def deterministic_recommend(model: dict, features: dict, goods_type: str, prices, cost: CostSpec,
                            alpha: float = 0.1) -> dict:
    level0 = model["conformal_level"] if model.get("conformal_level") is not None else alpha
    level0 = min(level0, alpha)
    cons = model.get("conservative", 0.1)
    rows = []
    for p in sorted(set(float(x) for x in prices)):
        if p <= cost.unit_cost:
            continue
        mu, sg = member_params(model, features, goods_type, p)
        rho = critical_ratio(p, cost)
        lvl = min(level0, rho)
        qb = np.maximum(np.expm1(mu + sg * norm.ppf(lvl)), 0)
        q = float(np.floor(qb[0] if len(qb) == 1 else np.quantile(qb, cons)))
        if cost.max_qty is not None:
            q = min(q, cost.max_qty)
        if q < cost.moq:
            q = 0.0
        sold = _expected_sold(mu, sg, q) if q > 0 else 0.0
        prof = (p * sold + cost.salvage * (q - sold) - cost.unit_cost * q - cost.fixed_cost) if q > 0 else 0.0
        sf_b = norm.sf((math.log1p(q) - mu) / sg) if q > 0 else np.ones(len(mu))
        q_nv = _mixture_quantile(mu, sg, rho)
        sold_nv = _expected_sold(mu, sg, q_nv)
        rows.append({
            "price": p, "safe_qty": int(q),
            "sellout_prob": float(sf_b.mean()) if q > 0 else None,
            "expected_sold": sold, "expected_leftover": q - sold if q > 0 else 0.0,
            "expected_profit": prof, "model_agreement": float((sf_b >= 1 - alpha).mean()),
            "demand_p10": _mixture_quantile(mu, sg, 0.1), "demand_p50": _mixture_quantile(mu, sg, 0.5),
            "demand_p90": _mixture_quantile(mu, sg, 0.9),
            "newsvendor_qty": int(q_nv),
            "newsvendor_sellout_prob": float(norm.sf((math.log1p(q_nv) - mu) / sg).mean()),
            "newsvendor_profit": (p * sold_nv + cost.salvage * (q_nv - sold_nv) - cost.unit_cost * q_nv
                                  - cost.fixed_cost) if q_nv >= cost.moq else 0.0,
        })
    ok = [r for r in rows if r["safe_qty"] > 0 and r["expected_profit"] > 0]
    best = max(ok, key=lambda r: r["expected_profit"]) if ok else None
    return {"rows": rows, "best": best, "level": level0}


# --------------------------------------------------------------------------- 올리기
def _upsert_model(session: Session, kind: str, payload: dict) -> None:
    from salesopt.storage import PublishedModel
    cur = session.get(PublishedModel, kind)
    now = datetime.now(UTC).replace(tzinfo=None)
    if cur is None:
        session.add(PublishedModel(kind=kind, trained_at=now, payload=payload))
    else:
        cur.trained_at, cur.payload = now, payload


def demo_model(seed: int = 0, n: int = 1500) -> dict:
    """가짜 데이터(simulate.py)로 학습한 예시 모델. 실제 판매 기록이 모이기 전 화면 확인용."""
    pred = fit_predictor(Simulator(seed=seed).records(n), NUMERIC_FEATURES, n_members=10, seed=seed)
    return export_model(pred, "demo", {"n_records": n, "source": "simulation"})


def real_model(session: Session, group_map, alpha: float = 0.1, calib_frac: float = 0.3, seed: int = 0,
               min_coverage: float = 0.5, min_elasticity: float = 0.0) -> dict | None:
    """so_sales_record(실제) → 특징 붙이기 → 학습 (+ conformal). 기록이 부족하면 None."""
    from salesopt.conformal import LEVEL_GRID, LevelConformal, min_records
    from salesopt.features import build_training_frame, feature_coverage
    from salesopt.storage import SalesRecord
    recs = session.scalars(select(SalesRecord).where(SalesRecord.source == "real")).all()
    if len(recs) < MIN_REAL_RECORDS:
        return None
    sales = pd.DataFrame([{c: getattr(r, c) for c in ("group_key", "goods_type", "product_name", "price",
                                                      "quantity_made", "quantity_sold", "sold_out", "sale_start")}
                          for r in recs])
    data = build_training_frame(session, sales, group_map)
    cov = feature_coverage(data)
    numeric = [f for f in NUMERIC_FEATURES if cov[f] >= min_coverage]
    n_cal = int(len(data) * calib_frac)
    level = None
    kw = {"min_elasticity": min_elasticity} if min_elasticity else {}
    if n_cal >= min_records(alpha):
        perm = np.random.default_rng(seed).permutation(len(data))
        cal, train = data.iloc[perm[:n_cal]], data.iloc[perm[n_cal:]]
        pred = fit_predictor(train, numeric, n_members=10, seed=seed, **kw)
        L = pred.lower_bounds(cal, cal.price.to_numpy(), LEVEL_GRID, seed=seed)
        level = LevelConformal(alpha).fit(L, cal.quantity_sold, cal.sold_out).level
    else:
        pred = fit_predictor(data, numeric, n_members=10, seed=seed, **kw)
    return export_model(pred, "real", {"n_records": len(data), "n_sold_out": int(data.sold_out.sum()),
                                       "n_calibration": n_cal if level is not None else 0, "alpha": alpha,
                                       "dropped_features": [f for f in NUMERIC_FEATURES if f not in numeric]},
                        conformal_level=level)


def publish_contexts(session: Session, group_map, goods_types, today: date | None = None) -> int:
    from salesopt.features import GroupInfo, build_context
    from salesopt.storage import IdolSnapshot, PublishedContext
    before = (today or date.today()) + timedelta(days=1)
    names = dict(session.execute(select(IdolSnapshot.group_key, IdolSnapshot.name)
                                 .order_by(IdolSnapshot.date)).all())
    keys = sorted(set(group_map) | set(names))
    session.execute(delete(PublishedContext))
    cache: dict = {}
    n = 0
    for k in keys:
        g = group_map.get(k) or GroupInfo(k, None, [])
        aliases = sorted({k, names.get(k, k), *g.extra_aliases})
        for gt in goods_types:
            ctx = build_context(session, g, gt, before, cache).iloc[0]
            feats = {f: (None if pd.isna(ctx[f]) else float(ctx[f])) for f in NUMERIC_FEATURES}
            feats["idol_snapshot_date"] = str(ctx["idol_snapshot_date"]) if ctx["idol_snapshot_date"] else None
            session.add(PublishedContext(group_key=k, group_name=names.get(k, k), aliases=aliases, goods_type=gt,
                                         as_of=before - timedelta(days=1), features=feats))
            n += 1
    return n


def publish(session: Session, group_map, alpha: float = 0.1, min_elasticity: float = 0.0) -> dict:
    from salesopt.sources.goods import hot_scores
    from salesopt.storage import PublishedModel
    out = {}
    if session.get(PublishedModel, "demo") is None:
        _upsert_model(session, "demo", demo_model())
        out["demo"] = "created"
    real = real_model(session, group_map, alpha=alpha, min_elasticity=min_elasticity)
    if real is not None:
        _upsert_model(session, "real", real)
        out["real"] = real["meta"]
    else:
        out["real"] = f"판매 기록 부족 (최소 {MIN_REAL_RECORDS}건)"
    types = set(GOODS_TYPES)
    hot = hot_scores(session, date.today())
    if not hot.empty:
        types |= set(hot.index)
    for kind in ("real", "demo"):
        m = session.get(PublishedModel, kind)
        if m is not None:
            types |= set(m.payload["categories"])
    out["contexts"] = publish_contexts(session, group_map, sorted(types))
    session.flush()
    return out
