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
# 두 모델을 같은 형태로 다룬다: 멤버마다 '정규분포 여러 개의 혼합' (log(1+D) 공간)
#   lognormal : 멤버당 정규분포 1개 (μ, σ)
#   cvae      : 멤버당 정규분포 Z_NODES 개 — 잠재변수 z 를 2차원 가우스-에르미트 대표점으로 고정하고
#               점마다 디코더(신경망)를 통과시켜 (μ_k, σ_k), 가중치 w_k
CVAE_MIN_RECORDS = 100          # 이 이상이면 딥러닝(cVAE)도 학습해 검증 기록으로 비교
GH_N = 8                        # 차원당 대표점 수 → 8 × 8 = 64 개


def _gh_nodes(n: int = GH_N) -> tuple[list, list]:
    from numpy.polynomial.hermite_e import hermegauss
    x, w = hermegauss(n)                      # 표준정규 기준 대표점·가중치
    w = w / w.sum()
    zz = [[float(a), float(b)] for a in x for b in x]
    ww = [float(wa * wb) for wa in w for wb in w]
    return zz, ww


def export_model(pred, kind: str, meta: dict | None = None, conformal_level: float | None = None,
                 conservative: float = 0.1) -> dict:
    from salesopt.models.cvae import CVAEDemand
    enc = pred.encoder
    members, model_type = [], None
    for m in pred.members:
        if isinstance(m, LogNormalDemand):
            model_type = "lognormal"
            b0, b, th, a0, a = m._unpack(m.w_)
            members.append({"b0": float(b0), "b": b.tolist(), "a0": float(a0), "a": a.tolist(),
                            "elasticity": m.elasticity().tolist(), "lp_center": m.lp_center_.tolist()})
        elif isinstance(m, CVAEDemand):
            model_type = "cvae"
            dec = m.net.dec
            layers = [[dec[i].weight.detach().cpu().numpy().tolist(), dec[i].bias.detach().cpu().numpy().tolist()]
                      for i in (0, 2, 4)]
            members.append({"layers": layers, "lp_m": m.lp_m, "lp_s": m.lp_s, "y_m": m.y_m, "y_s": m.y_s})
        else:
            raise ValueError(f"웹에 올릴 수 없는 모델: {type(m).__name__}")
    out = {
        "version": 2, "kind": kind, "model_type": model_type, "numeric": list(enc.numeric),
        "mean": enc.mean_.tolist(), "std": enc.std_.tolist(), "categories": list(enc.categories_),
        "members": members, "conservative": conservative, "conformal_level": conformal_level,
        "price_support": {k: list(v) for k, v in pred.price_support.items()}, "meta": meta or {},
    }
    if model_type == "cvae":
        out["z_nodes"], out["z_weights"] = _gh_nodes()
    return out


def _encode(model: dict, features: dict, goods_type: str) -> tuple[np.ndarray, int | None]:
    """특징 인코딩. 학습하지 않은 굿즈 종류면 원-핫을 0 으로 두고 g=None."""
    vals = np.array([np.nan if features.get(f) is None else float(features[f]) for f in model["numeric"]])
    num = np.nan_to_num((vals - np.array(model["mean"])) / np.array(model["std"]), nan=0.0)
    cats = model["categories"]
    onehot = np.zeros(len(cats) - 1)
    g = cats.index(goods_type) if goods_type in cats else None
    if g:
        onehot[g - 1] = 1.0
    return np.concatenate([num, onehot]), g


def supports(model: dict, goods_type: str) -> bool:
    """cVAE 는 학습한 굿즈 종류만 계산한다 (기록 없는 종류는 단순 모델이 '평균 종류'로 계산)."""
    return model.get("model_type", "lognormal") == "lognormal" or goods_type in model["categories"]


def _silu(v):
    return v / (1.0 + np.exp(-v))


def _lognormal_member(model, m, x, g, price, features):
    d = len(model["numeric"])
    b, a = np.array(m["b"]), np.array(m["a"])
    if g is not None:
        mu = m["b0"] + x @ b - m["elasticity"][g] * (math.log(price) - m["lp_center"][g])
        return np.array([mu]), np.array([math.exp(min(max(m["a0"] + x @ a, -4.0), 3.0))]), np.array([1.0])
    # 판매 기록 없는 종류 → 평균 굿즈 종류 (종류 효과 평균, 탄력성 평균, 불확실성 + 종류 간 분산)
    tb, ta = np.concatenate([[0.0], b[d:]]), np.concatenate([[0.0], a[d:]])
    ref = features.get("_price_ref")
    center = math.log(ref) if ref else float(np.mean(m["lp_center"]))
    mu = m["b0"] + x[:d] @ b[:d] + tb.mean() - float(np.mean(m["elasticity"])) * (math.log(price) - center)
    s0 = math.exp(min(max(m["a0"] + x[:d] @ a[:d] + ta.mean(), -4.0), 3.0))
    return np.array([mu]), np.array([math.sqrt(s0 ** 2 + tb.var())]), np.array([1.0])


def _cvae_member(model, m, x, price):
    """디코더 신경망을 대표점 64개에 통과 → 정규분포 64개의 혼합 (log(1+D) 공간)."""
    z = np.array(model["z_nodes"])
    lp = (math.log(price) - m["lp_m"]) / m["lp_s"]
    h = np.hstack([np.repeat(x[None, :], len(z), 0), np.full((len(z), 1), lp), z])
    (W1, b1), (W2, b2), (W3, b3) = [(np.array(W), np.array(bb)) for W, bb in m["layers"]]
    h = _silu(h @ W1.T + b1)
    h = _silu(h @ W2.T + b2)
    out = h @ W3.T + b3
    mu = out[:, 0] * m["y_s"] + m["y_m"]
    sg = np.exp(np.clip(out[:, 1], -5, 2)) * m["y_s"]
    return mu, sg, np.array(model["z_weights"])


def member_mixtures(model: dict, features: dict, goods_type: str, price: float) -> list:
    """멤버별 (μ_k, σ_k, w_k) 배열 목록."""
    x, g = _encode(model, features, goods_type)
    if model.get("model_type", "lognormal") == "cvae":
        if g is None:
            raise KeyError(goods_type)
        return [_cvae_member(model, m, x, price) for m in model["members"]]
    return [_lognormal_member(model, m, x, g, price, features) for m in model["members"]]


def member_params(model: dict, features: dict, goods_type: str, price: float) -> tuple[np.ndarray, np.ndarray]:
    """(단순 모델 전용) 멤버별 μ, σ."""
    mix = member_mixtures(model, features, goods_type, price)
    return np.array([m[0][0] for m in mix]), np.array([m[1][0] for m in mix])


def _mix_cdf(mix, y: float) -> float:
    return float(np.mean([(w * norm.cdf((y - mu) / sg)).sum() for mu, sg, w in mix]))


def _mix_sf_by_member(mix, q: float) -> np.ndarray:
    y = math.log1p(q)
    return np.array([(w * norm.sf((y - mu) / sg)).sum() for mu, sg, w in mix])


def _mixture_quantile(mix, level: float) -> float:
    """혼합분포(멤버 평균)의 수요 level 분위수 (이분법 80회)."""
    lo = min(float((mu - 9 * sg).min()) for mu, sg, _ in mix)
    hi = max(float((mu + 9 * sg).max()) for mu, sg, _ in mix)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _mix_cdf(mix, mid) < level:
            lo = mid
        else:
            hi = mid
    return max(math.floor(math.expm1(hi)), 0)


def _member_quantile(comp, level: float) -> float:
    """멤버 하나의 수요 level 분위수 (연속값). 정규분포 1개면 정확한 식, 여러 개면 이분법."""
    mu, sg, w = comp
    if len(mu) == 1:
        return max(math.expm1(mu[0] + sg[0] * norm.ppf(level)), 0.0)
    lo, hi = float((mu - 9 * sg).min()), float((mu + 9 * sg).max())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (w * norm.cdf((mid - mu) / sg)).sum() < level:
            lo = mid
        else:
            hi = mid
    return max(math.expm1(hi), 0.0)


def _expected_sold(mix, q: float) -> float:
    tot = 0.0
    for mu, sg, w in mix:
        d = np.maximum(np.floor(np.expm1(mu[:, None] + sg[:, None] * Z_GRID[None, :])), 0)
        tot += float((w[:, None] * np.minimum(q, d) * Z_W[None, :]).sum())
    return tot / len(mix)


def deterministic_recommend(model: dict, features: dict, goods_type: str, prices, cost: CostSpec,
                            alpha: float = 0.1) -> dict:
    level0 = model["conformal_level"] if model.get("conformal_level") is not None else alpha
    level0 = min(level0, alpha)
    cons = model.get("conservative", 0.1)
    rows = []
    for p in sorted(set(float(x) for x in prices)):
        if p <= cost.unit_cost:
            continue
        mix = member_mixtures(model, features, goods_type, p)
        rho = critical_ratio(p, cost)
        lvl = min(level0, rho)
        qb = np.array([_member_quantile(c, lvl) for c in mix])
        q = float(np.floor(qb[0] if len(qb) == 1 else np.quantile(qb, cons)))
        if cost.max_qty is not None:
            q = min(q, cost.max_qty)
        if q < cost.moq:
            q = 0.0
        sold = _expected_sold(mix, q) if q > 0 else 0.0
        prof = (p * sold + cost.salvage * (q - sold) - cost.unit_cost * q - cost.fixed_cost) if q > 0 else 0.0
        sf_b = _mix_sf_by_member(mix, q) if q > 0 else np.ones(len(mix))
        q_nv = _mixture_quantile(mix, rho)
        sold_nv = _expected_sold(mix, q_nv)
        rows.append({
            "price": p, "safe_qty": int(q),
            "sellout_prob": float(sf_b.mean()) if q > 0 else None,
            "expected_sold": sold, "expected_leftover": q - sold if q > 0 else 0.0,
            "expected_profit": prof, "model_agreement": float((sf_b >= 1 - alpha).mean()),
            "demand_p10": _mixture_quantile(mix, 0.1), "demand_p50": _mixture_quantile(mix, 0.5),
            "demand_p90": _mixture_quantile(mix, 0.9),
            "newsvendor_qty": int(q_nv),
            "newsvendor_sellout_prob": float(_mix_sf_by_member(mix, q_nv).mean()),
            "newsvendor_profit": (p * sold_nv + cost.salvage * (q_nv - sold_nv) - cost.unit_cost * q_nv
                                  - cost.fixed_cost) if q_nv >= cost.moq else 0.0,
        })
    ok = [r for r in rows if r["safe_qty"] > 0 and r["expected_profit"] > 0]
    best = max(ok, key=lambda r: r["expected_profit"]) if ok else None
    return {"rows": rows, "best": best, "level": level0, "known_type": goods_type in model["categories"]}


def log_score(model: dict, records: pd.DataFrame) -> float:
    """검증 기록에서 모델이 실제 결과에 준 평균 로그 확률 (클수록 잘 맞음). 완판 기록은 '그 이상일 확률'.
    두 모델(단순·딥러닝)을 같은 기준으로 비교하는 데 쓴다."""
    tot = 0.0
    for _, r in records.iterrows():
        feats = {f: (None if pd.isna(r[f]) else float(r[f])) for f in model["numeric"]}
        mix = member_mixtures(model, feats, r.goods_type, float(r.price))
        y = math.log1p(float(r.quantity_sold))
        if bool(r.sold_out):
            p = np.mean([(w * norm.sf((y - mu) / sg)).sum() for mu, sg, w in mix])
        else:      # 정수 판매량 d 의 확률 ≈ log(1+d) 밀도 × 1/(1+d)
            p = np.mean([(w * norm.pdf((y - mu) / sg) / sg).sum() for mu, sg, w in mix]) / (1 + r.quantity_sold)
        tot += math.log(max(p, 1e-300))
    return tot / max(len(records), 1)


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


def real_models(session: Session, group_map, alpha: float = 0.1, calib_frac: float = 0.3, seed: int = 0,
                min_coverage: float = 0.5, min_elasticity: float = 0.0) -> dict:
    """so_sales_record(실제) → 특징 붙이기 → 학습 + conformal 보정.

    반환: {"real_lognormal": …, "real_cvae": …(100건 이상일 때), "real": 자동 선택된 쪽}. 기록이 부족하면 {}.
      - 20건 이상: 단순 모델(로그정규 앙상블)
      - 100건 이상: 20% 를 비교용으로 떼어 두고 딥러닝(cVAE)도 학습 → 비교용 기록의 평균 로그 확률이
        더 큰 쪽을 '자동' 으로 쓴다 (건수가 아니라 실제로 더 잘 맞는지로 정함)
      - 보정: 남은 기록의 30% 를 떼어 모델별로 conformal 보정 (9건 이상일 때)
    """
    from salesopt.conformal import LEVEL_GRID, LevelConformal, min_records
    from salesopt.features import build_training_frame, feature_coverage
    from salesopt.pretrain import load_prior
    from salesopt.storage import SalesRecord
    recs = session.scalars(select(SalesRecord).where(SalesRecord.source == "real")).all()
    if len(recs) < MIN_REAL_RECORDS:
        return {}
    sales = pd.DataFrame([{c: getattr(r, c) for c in ("group_key", "goods_type", "product_name", "price",
                                                      "quantity_made", "quantity_sold", "sold_out", "sale_start")}
                          for r in recs])
    data = build_training_frame(session, sales, group_map).reset_index(drop=True)
    cov = feature_coverage(data)
    numeric = [f for f in NUMERIC_FEATURES if cov[f] >= min_coverage]
    perm = np.random.default_rng(seed).permutation(len(data))
    use_cvae = len(data) >= CVAE_MIN_RECORDS
    if use_cvae:
        try:
            import torch  # noqa: F401
        except ImportError:
            use_cvae = False
    n_val = int(len(data) * 0.2) if use_cvae else 0
    val, rest = data.iloc[perm[:n_val]], data.iloc[perm[n_val:]]
    n_cal = int(len(rest) * calib_frac)
    if n_cal < min_records(alpha):
        n_cal = 0
    cal, train = rest.iloc[:n_cal], rest.iloc[n_cal:]

    prior = load_prior()
    ln_kw = {"min_elasticity": min_elasticity} if min_elasticity else {}
    if prior:                                   # 공개 판매 데이터로 만든 가격 탄력성 사전분포
        ln_kw |= {"elasticity_prior": prior.elasticity_mean, "elasticity_sd": prior.elasticity_sd}
    base_meta = {"n_records": len(data), "n_sold_out": int(data.sold_out.sum()), "n_calibration": n_cal,
                 "n_validation": n_val, "alpha": alpha,
                 "dropped_features": [f for f in NUMERIC_FEATURES if f not in numeric]}

    def build(kind: str, **kw) -> dict:
        pred = fit_predictor(train, numeric, kind=kind, n_members=10 if kind == "lognormal" else 5, seed=seed, **kw)
        level = None
        if n_cal:
            L = pred.lower_bounds(cal, cal.price.to_numpy(), LEVEL_GRID, seed=seed)
            level = LevelConformal(alpha).fit(L, cal.quantity_sold, cal.sold_out).level
        meta = base_meta | {"model": kind}
        if kind == "lognormal":
            meta["elasticity_prior"] = [prior.elasticity_mean, prior.elasticity_sd] if prior else None
        return export_model(pred, "real", meta, conformal_level=level)

    out = {"real_lognormal": build("lognormal", **ln_kw)}
    if use_cvae:
        out["real_cvae"] = build("cvae")
        scores = {k: log_score(out[k], val) for k in ("real_lognormal", "real_cvae")}
        for k, v in scores.items():
            out[k]["meta"]["validation_log_score"] = round(v, 4)
        winner = max(scores, key=scores.get)
    else:
        winner = "real_lognormal"
    out["real"] = out[winner] | {"meta": out[winner]["meta"] | {"selected": winner.removeprefix("real_")}}
    return out


def publish_contexts(session: Session, group_map, goods_types, today: date | None = None) -> int:
    from salesopt.features import GroupInfo, build_context
    from salesopt.storage import IdolSnapshot, PublishedContext
    before = (today or date.today()) + timedelta(days=1)
    names = dict(session.execute(select(IdolSnapshot.group_key, IdolSnapshot.name)
                                 .order_by(IdolSnapshot.date)).all())
    keys = sorted(set(group_map) | set(names))
    from salesopt.sources.goods import goods_price_stats
    prices = goods_price_stats(session, before - timedelta(days=1))
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
            if gt in prices.index:                    # Goods 가 모은 이 종류의 판매가 분포
                pr = prices.loc[gt]
                feats |= {"_price_ref": round(float(pr.ref)), "_price_lo": round(float(pr.lo)),
                          "_price_hi": round(float(pr.hi)), "_price_n": int(pr.n),
                          "_price_source": pr.source}
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
    from salesopt.storage import PublishedModel as _PM
    models = real_models(session, group_map, alpha=alpha, min_elasticity=min_elasticity)
    for kind in ("real", "real_lognormal", "real_cvae"):
        if kind in models:
            _upsert_model(session, kind, models[kind])
        elif (old := session.get(_PM, kind)) is not None:      # 이번에 만들지 않은 모델은 지운다
            session.delete(old)
    out["real"] = (models["real"]["meta"] if models else f"판매 기록 부족 (최소 {MIN_REAL_RECORDS}건)")
    types = set(GOODS_TYPES)
    hot = hot_scores(session, date.today())
    if not hot.empty:
        types |= set(hot.index)
    from salesopt.sources.goods import goods_price_stats
    types |= set(goods_price_stats(session, date.today()).index)
    for kind in ("real", "demo"):
        m = session.get(PublishedModel, kind)
        if m is not None:
            types |= set(m.payload["categories"])
    out["contexts"] = publish_contexts(session, group_map, sorted(types))
    prices = goods_price_stats(session, date.today())          # 굿즈 종류별 가격 자료 출처 요약 (로그 확인용)
    src = prices["source"].value_counts().to_dict() if not prices.empty else {}
    out["price_data"] = {"판매가": src.get("retail", 0), "KREAM 환산": src.get("resale", 0),
                         "없음(원가 기준)": len(types) - len(prices)}
    session.flush()
    return out
