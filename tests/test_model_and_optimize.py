import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from salesopt.conformal import LEVEL_GRID, LevelConformal, min_records
from salesopt.models import fit_predictor
from salesopt.models.lognormal import LogNormalDemand
from salesopt.optimize import CostSpec, critical_ratio, profit, recommend
from salesopt.simulate import NUMERIC_FEATURES, Simulator, default_cost


class KnownLogNormal:
    """참 분포를 그대로 내는 가짜 예측기 (최적화 로직만 검증)."""
    def __init__(self, mu0=7.0, el=1.5, sigma=0.5, ref=10000.0, B=1):
        self.mu0, self.el, self.sigma, self.ref, self.B = mu0, el, sigma, ref, B

    def mu(self, p):
        return self.mu0 - self.el * np.log(np.asarray(p) / self.ref)

    def sample_members(self, context, prices, m, rng):
        mu = self.mu(prices)
        y = mu[None, :, None] + self.sigma * rng.standard_normal((self.B, len(prices), m))
        return np.floor(np.expm1(y)).clip(min=0)


def test_profit_and_critical_ratio():
    c = CostSpec(unit_cost=4000, salvage=1000, fixed_cost=100)
    d = np.array([0, 50, 200])
    assert profit(d, 10000, 100, c).tolist() == [
        1000 * 100 - 4000 * 100 - 100,
        10000 * 50 + 1000 * 50 - 4000 * 100 - 100,
        10000 * 100 - 4000 * 100 - 100,
    ]
    assert profit(d, 10000, 0, c).tolist() == [0, 0, 0]
    assert critical_ratio(10000, c) == pytest.approx(6000 / 9000)


def test_recommend_safe_quantity_matches_true_quantile():
    pred = KnownLogNormal()
    cost = CostSpec(unit_cost=3000, moq=10)
    rec = recommend(pred, pd.DataFrame([{"x": 1}]), [8000, 10000, 12000], cost, alpha=0.1, m=40000)
    for _, r in rec.table.iterrows():
        true_q = np.expm1(pred.mu(r.price) + pred.sigma * norm.ppf(0.1))
        assert r.safe_qty == pytest.approx(true_q, rel=0.03)
        assert r.sellout_prob >= 0.89
        assert r.newsvendor_sellout_prob < r.sellout_prob        # 논문 원래 해는 완판 확률이 더 낮다
    assert rec.best["price"] == rec.table.loc[rec.table.expected_profit.idxmax(), "price"]


def test_recommend_no_profitable_price():
    rec = recommend(KnownLogNormal(mu0=2.0), pd.DataFrame([{"x": 1}]), [8000, 10000],
                    CostSpec(unit_cost=3000, fixed_cost=10_000_000, moq=1), alpha=0.1, m=2000)
    assert rec.best is None and rec.notes


def test_lognormal_handles_censoring():
    """완판(중도절단) 기록을 판매량 그대로 쓰면 수요를 과소추정한다 → 생존확률 우도로 바로잡는지."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 2))
    lp = np.log(rng.uniform(8000, 14000, n))
    g = np.zeros(n, int)
    mu = 6 + 0.5 * X[:, 0] - 1.2 * (lp - lp.mean())
    d = np.expm1(mu + 0.5 * rng.standard_normal(n))
    made = np.expm1(mu + rng.normal(0, 0.5, n))
    sold, cens = np.minimum(d, made), d >= made
    m = LogNormalDemand(ridge=0.01).fit(X, lp, sold, cens, g)
    mu_hat, s_hat = m.params(X, lp, g)
    assert np.mean(mu_hat - mu) == pytest.approx(0, abs=0.05)
    assert np.mean(s_hat) == pytest.approx(0.5, abs=0.05)
    assert m.elasticity()[0] == pytest.approx(1.2, abs=0.15)
    naive = LogNormalDemand(ridge=0.01).fit(X, lp, sold, np.zeros(n, bool), g)
    assert np.mean(naive.params(X, lp, g)[0] - mu) < -0.1          # 무시하면 과소추정


def test_conformal_level():
    rng = np.random.default_rng(1)
    n = 400
    levels = LEVEL_GRID
    mu = rng.normal(6, 0.5, n)
    d = np.expm1(mu + 0.5 * rng.standard_normal(n))
    calibrated = np.expm1(mu[:, None] + 0.5 * norm.ppf(levels)[None, :])
    lvl = LevelConformal(0.1).fit(calibrated, d, np.zeros(n, bool), levels).level
    assert 0.07 <= lvl <= 0.1
    optimistic = np.expm1(mu[:, None] + 0.3 * norm.ppf(levels)[None, :] + 0.2)   # 하한을 너무 높게 잡는 모델
    lvl2 = LevelConformal(0.1).fit(optimistic, d, np.zeros(n, bool), levels).level
    assert lvl2 < 0.05
    with pytest.raises(ValueError):
        LevelConformal(0.1).fit(calibrated[:5], d[:5], np.zeros(5, bool), levels)
    assert min_records(0.1) == 9 and min_records(0.05) == 19


def test_conformal_censoring_modes():
    rng = np.random.default_rng(2)
    n = 300
    mu = rng.normal(6, 0.5, n)
    d = np.expm1(mu + 0.5 * rng.standard_normal(n))
    made = np.expm1(mu + rng.normal(-0.3, 0.5, n))
    sold, cens = np.minimum(d, made), d >= made
    L = np.expm1(mu[:, None] + 0.5 * norm.ppf(LEVEL_GRID)[None, :])
    strict = LevelConformal(0.1, censored="strict").fit(L, sold, cens).level
    imp = LevelConformal(0.1, censored="impute").fit(L, sold, cens).level
    assert strict <= imp <= 0.1 + 1e-9                    # strict 가 가장 보수적


def test_end_to_end_simulation_is_safe():
    sim = Simulator(seed=3)
    train = sim.records(400)
    pred = fit_predictor(train, NUMERIC_FEATURES, n_members=5, seed=3)
    ctxs = sim.contexts(15)
    probs = []
    for i in range(len(ctxs)):
        ctx = ctxs.iloc[[i]].reset_index(drop=True)
        gt = ctx.goods_type[0]
        ref = {"키링": 12000, "인형": 32000, "아크릴스탠드": 18000, "응원봉": 45000, "티셔츠": 35000,
               "포토카드": 8000}[gt]
        rec = recommend(pred, ctx, ref * np.linspace(0.8, 1.3, 6), default_cost(gt), alpha=0.1, m=2000)
        if rec.best:
            probs.append(sim.true_sellout_prob(ctx, rec.best["price"], rec.best["safe_qty"])[0])
    assert len(probs) >= 8
    assert np.mean(probs) >= 0.85


def test_unknown_goods_type_raises():
    sim = Simulator(seed=0)
    pred = fit_predictor(sim.records(80), NUMERIC_FEATURES, n_members=1)
    ctx = sim.contexts(1).assign(goods_type="우산")
    with pytest.raises(ValueError):
        pred.sample_members(ctx, [10000], 10, np.random.default_rng(0))


def test_cvae_smoke():
    pytest.importorskip("torch")
    sim = Simulator(seed=4)
    train = sim.records(300)
    pred = fit_predictor(train, NUMERIC_FEATURES, kind="cvae", n_members=2, epochs=30)
    ctx = sim.contexts(1, goods_type="키링")
    S = pred.sample_members(ctx, [10000, 14000], 500, np.random.default_rng(0))
    assert S.shape == (2, 2, 500) and (S >= 0).all()
    assert np.median(S[:, 0]) >= np.median(S[:, 1])          # 가격이 오르면 수요가 줄어야 함 (단조성 벌점)
