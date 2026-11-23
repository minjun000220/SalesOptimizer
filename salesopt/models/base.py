"""수요분포 모델 공통 틀.

논문의 생성기 G(x, p, η) 역할: 특징 x 와 가격 p 를 받아 수요 샘플을 원하는 만큼 만든다.
데이터 양에 따라 모델을 바꿔 끼울 수 있도록 같은 인터페이스를 쓴다.
  - LogNormalDemand : 데이터가 적을 때 (수십~수백 건). 파라미터 수가 적고 안정적
  - CVAEDemand      : 데이터가 많을 때 (1,000건 이상). 논문의 cDGM (조건부 VAE)

공통 처리
  - 수요는 log(1 + d) 로 다룬다 (0 이상, 꼬리가 긴 분포)
  - 완판 기록은 '수요 ≥ 판매량' 이라는 하한만 아는 중도절단 데이터 → 생존확률로 우도 계산
  - 가격이 오르면 수요가 줄도록 제약 (가격 탄력성 ≥ 0)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureEncoder:
    """숫자 특징 표준화 + 굿즈 종류 원-핫. 굿즈 종류 번호는 종류별 가격 탄력성에 쓴다.
    빈 값(NaN)은 학습 평균으로 채운다 (표준화 후 0)."""
    numeric: list[str]
    categorical: str = "goods_type"
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    categories_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        v = df[self.numeric].to_numpy(float)
        if np.isnan(v).all(axis=0).any():
            empty = [c for c, e in zip(self.numeric, np.isnan(v).all(axis=0)) if e]
            raise ValueError(f"값이 하나도 없는 특징: {empty} (특징 목록에서 빼세요)")
        self.mean_ = np.nanmean(v, axis=0)
        sd = np.nanstd(v, axis=0)
        self.std_ = np.where(sd > 1e-9, sd, 1.0)
        self.categories_ = sorted(df[self.categorical].astype(str).unique())
        return self

    def type_index(self, df: pd.DataFrame) -> np.ndarray:
        lookup = {c: i for i, c in enumerate(self.categories_)}
        idx = df[self.categorical].astype(str).map(lookup)
        if idx.isna().any():
            unknown = sorted(set(df[self.categorical][idx.isna()]))
            raise ValueError(f"학습 데이터에 없는 굿즈 종류: {unknown}")
        return idx.to_numpy(int)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = np.nan_to_num((df[self.numeric].to_numpy(float) - self.mean_) / self.std_, nan=0.0)  # 빈 값 = 평균
        t = self.type_index(df)
        onehot = np.zeros((len(df), max(len(self.categories_) - 1, 0)))
        rows = np.where(t > 0)[0]
        onehot[rows, t[rows] - 1] = 1.0                      # 첫 종류를 기준(절편)으로
        return np.hstack([num, onehot])


class DemandModel:
    """X: 인코딩된 특징 (n, d), log_price: (n,), groups: 굿즈 종류 번호 (n,)"""

    def fit(self, X, log_price, sold, censored, groups, n_groups=None) -> "DemandModel":
        raise NotImplementedError

    def sample(self, X, log_price, groups, m: int, rng: np.random.Generator) -> np.ndarray:
        """(n, m) 수요 샘플 (개수 단위, 0 이상)"""
        raise NotImplementedError


@dataclass
class DemandPredictor:
    """인코더 + 모델 B개(앙상블). optimize.recommend 와 conformal 이 사용한다."""
    encoder: FeatureEncoder
    members: list
    price_support: dict = field(default_factory=dict)     # 굿즈 종류 → (학습 가격 하한, 상한)

    def sample_members(self, context: pd.DataFrame, prices, m: int, rng) -> np.ndarray:
        """한 상황(1행)을 가격 J개로 복제해 샘플 → (B, J, m)"""
        prices = np.asarray(prices, dtype=float)
        rep = context.loc[context.index.repeat(len(prices))].reset_index(drop=True)
        X, g = self.encoder.transform(rep), self.encoder.type_index(rep)
        lp = np.log(prices)
        return np.stack([mem.sample(X, lp, g, m, rng) for mem in self.members])

    def samples_at(self, df: pd.DataFrame, prices, m: int, rng) -> np.ndarray:
        """여러 상황 각각의 가격에서 샘플 → (B, n, m)"""
        X, g = self.encoder.transform(df), self.encoder.type_index(df)
        lp = np.log(np.asarray(prices, dtype=float))
        return np.stack([mem.sample(X, lp, g, m, rng) for mem in self.members])

    def lower_bounds(self, df: pd.DataFrame, prices, levels, conservative: float = 0.1,
                     m: int = 2000, seed: int = 0) -> np.ndarray:
        """각 상황·각 분위수 수준에서 '안전 수량' 규칙과 같은 방식으로 계산한 값 → (n, A). conformal 보정용."""
        S = self.samples_at(df, prices, m, np.random.default_rng(seed))     # (B, n, m)
        qs = np.quantile(S, np.asarray(levels), axis=2)                       # (A, B, n)
        L = qs[:, 0, :] if S.shape[0] == 1 else np.quantile(qs, conservative, axis=1)   # (A, n)
        return np.floor(L.T)

    def price_range(self, goods_type: str) -> tuple[float, float] | None:
        return self.price_support.get(goods_type)


def price_support(df: pd.DataFrame, categorical: str = "goods_type", lo=0.05, hi=0.95) -> dict:
    """굿즈 종류별로 학습 데이터에 있었던 가격 범위. 이 밖의 가격은 외삽이라 믿기 어렵다."""
    out = {}
    for t, g in df.groupby(categorical):
        out[str(t)] = (float(g.price.quantile(lo)), float(g.price.quantile(hi)))
    return out
