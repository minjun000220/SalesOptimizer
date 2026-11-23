"""시뮬레이션 백테스트: 정답 수요분포를 아는 상태에서 추천이 정말 '안전'한지 확인한다.

정책
  safe        : 이 프로그램 (확률 제약 + 앙상블 보수화 [+ conformal])
  newsvendor  : 논문 원래 해 (기대이익 최대, ρ 분위수) — 안전 기준 없음
  oracle      : 참 분포를 알고 같은 확률 제약을 푼 해 (도달 가능한 최선)

지표
  sellout_mean   : 추천 (가격, 수량)에서 참 완판 확률의 평균         (목표: ≥ 1 − α)
  violation_rate : 참 완판 확률이 1 − α − tol 보다 낮은 상황의 비율  (작을수록 안전)
  profit_ratio   : 참 기대이익 합 / oracle 의 참 기대이익 합          (클수록 이익을 덜 포기)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from salesopt.conformal import LEVEL_GRID, LevelConformal
from salesopt.models import fit_predictor
from salesopt.optimize import CostSpec, critical_ratio, recommend
from salesopt.simulate import GOODS_TYPES, NUMERIC_FEATURES, Simulator, default_cost

PRICE_MULTS = np.linspace(0.75, 1.35, 13)


def price_grid(goods_type: str) -> np.ndarray:
    return np.round(GOODS_TYPES[goods_type]["ref_price"] * PRICE_MULTS / 500) * 500


def oracle(sim: Simulator, ctx: pd.DataFrame, grid, cost: CostSpec, alpha: float) -> tuple[float, float, float]:
    best = (np.nan, 0.0, 0.0)
    for p in grid:
        if p <= cost.unit_cost:
            continue
        level = min(alpha, critical_ratio(p, cost))
        q = float(sim.true_demand_quantile(ctx, p, level)[0])
        if q < cost.moq:
            continue
        pr = sim.true_expected_profit(ctx, p, q, cost)
        if pr > best[2]:
            best = (p, q, pr)
    return best


def run_backtest(n_train: int = 300, n_calib: int = 0, n_test: int = 100, alpha: float = 0.1,
                 kind: str = "lognormal", n_members: int = 10, conservative: float = 0.1,
                 confounding: bool = False, seed: int = 0, tol: float = 0.02, m: int = 3000,
                 censored: str = "strict",
                 model_kw: dict | None = None) -> dict:
    sim = Simulator(seed=seed, confounding=confounding)
    train = sim.records(n_train)
    pred = fit_predictor(train, NUMERIC_FEATURES, kind=kind, n_members=n_members, seed=seed, **(model_kw or {}))
    conf = None
    if n_calib:
        cal = sim.records(n_calib)
        L = pred.lower_bounds(cal, cal.price.to_numpy(), LEVEL_GRID, conservative, seed=seed + 1)
        conf = LevelConformal(alpha, censored=censored).fit(L, cal.quantity_sold.to_numpy(),
                                                            cal.sold_out.to_numpy())

    tests = sim.contexts(n_test)
    rows = []
    for i in range(n_test):
        ctx = tests.iloc[[i]].reset_index(drop=True)
        gt = ctx.goods_type[0]
        cost, grid = default_cost(gt), price_grid(gt)
        rec = recommend(pred, ctx, grid, cost, alpha=alpha, conservative=conservative, m=m,
                        conformal=conf, seed=seed + 100 + i)
        o_p, o_q, o_pr = oracle(sim, ctx, grid, cost, alpha)
        row = {"goods_type": gt, "oracle_profit": o_pr}
        if rec.best:
            p, q = rec.best["price"], rec.best["safe_qty"]
            row.update(safe_price=p, safe_qty=q, safe_sellout=float(sim.true_sellout_prob(ctx, p, q)[0]),
                       safe_profit=sim.true_expected_profit(ctx, p, q, cost))
        else:
            row.update(safe_price=np.nan, safe_qty=0, safe_sellout=1.0, safe_profit=0.0)
        t = rec.table
        if not t.empty:                       # 같은 모델로 논문 원래 기준 (기대이익 최대, 안전 제약 없음)
            r = t.loc[t.newsvendor_profit.idxmax()]
            p_nv, q_nv = float(r.price), float(r.newsvendor_qty) if r.newsvendor_qty >= cost.moq else 0.0
            row.update(nv_sellout=float(sim.true_sellout_prob(ctx, p_nv, q_nv)[0]),
                       nv_profit=sim.true_expected_profit(ctx, p_nv, q_nv, cost))
        rows.append(row)
    df = pd.DataFrame(rows)
    made = df.safe_qty > 0
    target = 1 - alpha
    return {
        "n_train": n_train, "n_calib": n_calib, "model": kind, "alpha": alpha, "confounding": confounding,
        "safe_sellout_mean": float(df.loc[made, "safe_sellout"].mean()) if made.any() else np.nan,
        "safe_violation_rate": float((df.loc[made, "safe_sellout"] < target - tol).mean()) if made.any() else 0.0,
        "safe_profit_ratio": float(df.safe_profit.sum() / df.oracle_profit.sum()),
        "safe_make_rate": float(made.mean()),
        "nv_sellout_mean": float(df.nv_sellout.mean()),
        "nv_violation_rate": float((df.nv_sellout < target - tol).mean()),
        "nv_profit_ratio": float(df.nv_profit.sum() / df.oracle_profit.sum()),
        "calibrated_level": conf.level if conf else None,
        "detail": df,
    }
