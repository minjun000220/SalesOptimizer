import numpy as np
import pandas as pd
import pytest

from salesopt.pretrain import combine, estimate


def _panel(e_mean=1.3, e_sd=0.4, units=300, weeks=40, seed=0, confound=True):
    """상품마다 탄력성이 다르고, '원래 잘 팔리는 상품이 비싸다' (상품 효과와 가격이 상관) + 계절 효과가 있는 패널."""
    rng = np.random.default_rng(seed)
    a = rng.normal(4, 1, units)                     # 상품 효과
    e = rng.normal(e_mean, e_sd, units)
    season = rng.normal(0, 0.3, weeks)
    rows = []
    for u in range(units):
        base_p = np.exp(2.3 + (0.5 * (a[u] - 4) if confound else 0) + rng.normal(0, 0.2))
        promo = rng.random(weeks) < 0.2
        lp = np.log(base_p) + np.where(promo, -0.2, 0) + rng.normal(0, 0.03, weeks)
        y = a[u] + season - e[u] * (lp - np.log(base_p)) + rng.normal(0, 0.3, weeks)
        q = np.round(np.expm1(y)).clip(min=0)
        rows.append(pd.DataFrame({"unit": u, "time": np.arange(weeks), "q": q, "p": np.exp(lp)}))
    return pd.concat(rows, ignore_index=True)


def test_estimate_recovers_elasticity_despite_confounding():
    df = _panel()
    est = estimate(df, "unit", "time", "sim")
    assert est.elasticity == pytest.approx(1.3, abs=0.08)
    assert est.spread == pytest.approx(0.4, abs=0.12)
    # 상품 효과를 지우지 않으면(단순 회귀) '비싼 상품이 잘 팔린다' 때문에 탄력성이 크게 틀린다
    naive = -np.polyfit(np.log(df.p), np.log1p(df.q), 1)[0]
    assert naive < 0.5


def test_combine_is_wide_when_sources_disagree():
    a = estimate(_panel(e_mean=1.6, seed=1), "unit", "time", "a")
    b = estimate(_panel(e_mean=1.0, seed=2), "unit", "time", "b")
    p = combine([a, b])
    assert 1.2 < p.elasticity_mean < 1.4
    assert p.elasticity_sd > max(a.spread, b.spread)        # 데이터 간 차이까지 반영해 넓게
