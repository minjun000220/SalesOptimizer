"""수요분포 모델과 학습 진입점."""
from __future__ import annotations

import numpy as np
import pandas as pd

from salesopt.models.base import DemandPredictor, FeatureEncoder, price_support
from salesopt.models.lognormal import LogNormalDemand

MODELS = ("lognormal", "cvae")


def _make(kind: str, seed: int, **kw):
    if kind == "lognormal":
        return LogNormalDemand(**kw)
    if kind == "cvae":
        from salesopt.models.cvae import CVAEDemand
        return CVAEDemand(seed=seed, **kw)
    raise ValueError(f"모델 종류: {MODELS}")


def fit_predictor(records: pd.DataFrame, numeric: list[str], kind: str = "lognormal", n_members: int = 10,
                  seed: int = 0, categorical: str = "goods_type", **model_kw) -> DemandPredictor:
    """판매 기록 → 앙상블 수요 예측기.

    records 에 필요한 열: numeric 특징, goods_type, price, quantity_sold, sold_out
    멤버마다 기록을 복원추출(bootstrap)해서 학습 → 멤버 간 차이 = 데이터가 적어서 생기는 모델 불확실성.
    n_members=1 이면 전체 데이터로 하나만 학습.
    """
    need = set(numeric) | {categorical, "price", "quantity_sold", "sold_out"}
    missing = need - set(records.columns)
    if missing:
        raise ValueError(f"판매 기록에 없는 열: {sorted(missing)}")
    records = records.reset_index(drop=True)
    enc = FeatureEncoder(numeric, categorical).fit(records)
    X, g = enc.transform(records), enc.type_index(records)
    lp = np.log(records.price.to_numpy(float))
    sold = records.quantity_sold.to_numpy(float)
    cens = records.sold_out.astype(bool).to_numpy()
    rng = np.random.default_rng(seed)
    members = []
    for b in range(n_members):
        idx = np.arange(len(records)) if n_members == 1 else rng.integers(0, len(records), len(records))
        members.append(_make(kind, seed + b, **model_kw).fit(X[idx], lp[idx], sold[idx], cens[idx], g[idx],
                                                         n_groups=len(enc.categories_)))
    return DemandPredictor(enc, members, price_support(records, categorical))
