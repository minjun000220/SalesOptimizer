"""정답을 아는 가짜 판매 데이터 (논문 6장의 시뮬레이션 역할).

실제 판매 데이터가 들어오기 전에 전체 파이프라인을 검증하는 용도다.
특징 열 이름은 실제 데이터(features.py)와 똑같이 맞춰서, 모델·최적화 코드를 바꾸지 않고 교체할 수 있다.

수요 모형 (판매 기간 T 동안의 수요 D):
    log(1 + D) ~ N( μ(x, p), σ(x)² )
    μ = 종류 기본값 + 그룹 인기 + 굿즈 화제성 + 비선형 상호작용 − 탄력성 · log(p / 기준가)
    σ = 논란이 많을수록, 호감도 변동이 클수록 불확실

과거 기록은 생산 수량 q 만큼만 팔 수 있으므로 판매량 = min(D, q), D ≥ q 이면 완판(중도절단).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from salesopt.optimize import CostSpec

NUMERIC_FEATURES = [
    # 아이돌 그룹 쪽 (IdolSentiments)
    "favor",               # 그룹 호감도 0~100
    "interest",            # 그룹 관심도 0~100
    "controversy",         # 논란도 0~1
    "log_mentions_per_day",
    "favor_change_7d",     # 최근 7일 호감도 변화
    "member_hhi",          # 멤버 언급 쏠림도 (Σ share²)
    # 굿즈 쪽 (Goods)
    "hot_index",           # 굿즈 종류 핫지수 0~100
    "buzz_z",              # 커뮤니티 언급 증가 z
    "sold_out_ratio",      # 최근 품절 비율
    "group_type_buzz_7d",  # 이 그룹 갤러리에서 이 굿즈 종류 언급 수 (7일)
]
CATEGORICAL = "goods_type"
HIDDEN_SHOCK = 0.3   # 특징에 없는 수요 충격의 크기 (log 단위)

GOODS_TYPES: dict[str, dict] = {
    "키링": dict(ref_price=12000, base=5.6, elasticity=1.6, extra_noise=0.00),
    "인형": dict(ref_price=32000, base=5.0, elasticity=1.2, extra_noise=0.10),
    "아크릴스탠드": dict(ref_price=18000, base=5.3, elasticity=1.4, extra_noise=0.05),
    "응원봉": dict(ref_price=45000, base=6.2, elasticity=0.7, extra_noise=0.00),
    "티셔츠": dict(ref_price=35000, base=4.6, elasticity=1.9, extra_noise=0.15),
    "포토카드": dict(ref_price=8000, base=6.0, elasticity=1.1, extra_noise=0.10),
}


def default_cost(goods_type: str) -> CostSpec:
    """시뮬레이션용 비용: 원가 = 기준가의 35%, 남은 재고는 원가의 20%에 처분, 1회 생산 고정비 30만 원."""
    ref = GOODS_TYPES[goods_type]["ref_price"]
    unit = 0.35 * ref
    return CostSpec(unit_cost=unit, salvage=0.2 * unit, fixed_cost=300_000, moq=50)


@dataclass
class Simulator:
    seed: int = 0
    confounding: bool = False    # True: 숨은 수요 충격 u 가 큰 곳에 가격을 높게 매긴 과거 기록 (내생성)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ 특징
    def contexts(self, n: int, goods_type: str | None = None) -> pd.DataFrame:
        r = self.rng
        types = list(GOODS_TYPES)
        df = pd.DataFrame({
            "favor": r.uniform(40, 80, n),
            "interest": r.uniform(30, 100, n),
            "controversy": 0.1 + 0.3 * r.beta(2, 4, n),
            "log_mentions_per_day": r.normal(6.0, 0.7, n),
            "favor_change_7d": r.normal(0, 3, n),
            "member_hhi": r.uniform(1 / 7, 0.5, n),
            "hot_index": r.uniform(20, 90, n),
            "buzz_z": r.normal(0, 1, n),
            "sold_out_ratio": r.beta(2, 5, n),
            "group_type_buzz_7d": r.poisson(np.exp(r.normal(2.0, 1.0, n))),
            CATEGORICAL: [goods_type] * n if goods_type else r.choice(types, n),
        })
        return df

    # ------------------------------------------------------------------ 참 수요분포
    @staticmethod
    def _mu0(df: pd.DataFrame) -> np.ndarray:
        """가격 효과를 뺀 평균."""
        base = df[CATEGORICAL].map(lambda t: GOODS_TYPES[t]["base"]).to_numpy()
        f = (df.favor.to_numpy() - 60) / 10
        h = (df.hot_index.to_numpy() - 50) / 20
        return (base
                + 0.8 * (df.log_mentions_per_day.to_numpy() - 6.0)
                + 0.35 * f
                + 0.30 * h
                + 0.25 * df.buzz_z.to_numpy()
                + 0.30 * (np.log1p(df.group_type_buzz_7d.to_numpy()) - 2.0)
                + 0.60 * (df.sold_out_ratio.to_numpy() - 0.3)
                + 0.20 * np.tanh(f) * h                     # 호감도 × 화제성 상호작용 (비선형)
                + 0.80 * (df.member_hhi.to_numpy() - 0.3))

    def mu_sigma(self, df: pd.DataFrame, price) -> tuple[np.ndarray, np.ndarray]:
        price = np.broadcast_to(np.asarray(price, dtype=float), (len(df),))
        ref = df[CATEGORICAL].map(lambda t: GOODS_TYPES[t]["ref_price"]).to_numpy()
        el = df[CATEGORICAL].map(lambda t: GOODS_TYPES[t]["elasticity"]).to_numpy()
        extra = df[CATEGORICAL].map(lambda t: GOODS_TYPES[t]["extra_noise"]).to_numpy()
        mu = self._mu0(df) - el * np.log(price / ref)
        sigma = 0.30 + 1.0 * df.controversy.to_numpy() + extra + 0.03 * np.abs(df.favor_change_7d.to_numpy())
        return mu, np.sqrt(sigma ** 2 + HIDDEN_SHOCK ** 2)      # 모델이 못 보는 수요 충격 u 까지 포함한 전체 불확실성

    def demand(self, df: pd.DataFrame, price) -> np.ndarray:
        mu, sigma = self.mu_sigma(df, price)
        return np.floor(np.expm1(mu + sigma * self.rng.standard_normal(len(df)))).clip(min=0)

    # ------------------------------------------------------------------ 과거 판매 기록
    def records(self, n: int) -> pd.DataFrame:
        df = self.contexts(n)
        ref = df[CATEGORICAL].map(lambda t: GOODS_TYPES[t]["ref_price"]).to_numpy()
        u = self.rng.standard_normal(n)              # 판매자는 알았지만 데이터에는 없는 수요 충격 (예: 미공개 컴백 일정)
        if self.confounding:                         # 내생성: 잘 팔릴 걸 알면 비싸게 매겼다
            mult = np.exp(0.30 * u + self.rng.normal(0, 0.10, n))
        else:
            mult = np.exp(self.rng.uniform(np.log(0.7), np.log(1.4), n))
        price = np.round(ref * mult.clip(0.6, 1.6) / 500) * 500
        mu, sig_all = self.mu_sigma(df, price)
        sigma = np.sqrt(sig_all ** 2 - HIDDEN_SHOCK ** 2)
        made = np.maximum(np.round(np.expm1(mu + HIDDEN_SHOCK * u + self.rng.normal(0, 0.5, n))), 30)  # 담당자의 감
        d = np.floor(np.expm1(mu + HIDDEN_SHOCK * u + sigma * self.rng.standard_normal(n))).clip(min=0)
        df["price"] = price
        df["quantity_made"] = made
        df["quantity_sold"] = np.minimum(d, made)
        df["sold_out"] = d >= made
        df["source"] = "sim"
        return df

    # ------------------------------------------------------------------ 평가용 정답
    def true_sellout_prob(self, df: pd.DataFrame, price, q) -> np.ndarray:
        """P(D ≥ q). floor 때문에 D ≥ q ⇔ log1p 값 ≥ log1p(q)."""
        mu, sigma = self.mu_sigma(df, price)
        q = np.asarray(q, dtype=float)
        return np.where(q <= 0, 1.0, norm.sf((np.log1p(q) - mu) / sigma))

    def true_demand_quantile(self, df: pd.DataFrame, price, level) -> np.ndarray:
        mu, sigma = self.mu_sigma(df, price)
        return np.floor(np.expm1(mu + sigma * norm.ppf(level))).clip(min=0)

    def true_expected_profit(self, row: pd.DataFrame, price: float, q: float, cost: CostSpec,
                             n_mc: int = 40000) -> float:
        mu, sigma = self.mu_sigma(row, price)
        z = np.random.default_rng(12345).standard_normal(n_mc)     # 공통 난수: 정책 간 비교를 공정하게
        d = np.floor(np.expm1(mu[0] + sigma[0] * z)).clip(min=0)
        from salesopt.optimize import profit
        return float(profit(d, price, q, cost).mean())
