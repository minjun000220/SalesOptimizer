"""Conformal 보정: 실제 판매 기록 n건으로 '완판 확률 ≥ 1 − α' 가 지켜지도록 분위수 수준을 낮춘다.

방법 (분위수 수준 보정, distributional conformal):
  안전 수량 규칙 L(a) = '수요의 a 분위수' (앙상블 보수화 포함) 는 a 에 대해 증가한다.
  보정 기록 i 의 점수 sᵢ = max{ a : L_i(a) ≤ Dᵢ }  (= 모델 기준으로 실제 수요가 몇 분위였나, PIT)
  새 상황에서 D ≥ L(a) ⇔ s ≥ a 이므로,
      a* = s 들 중 ⌊(n+1)α⌋ 번째로 작은 값   →   P(D < L(a*)) ≤ α   (교환가능성만 가정, 모델이 틀려도 성립)
  추천할 때 α 대신 a* 수준의 분위수를 쓴다. 모델이 낙관적이면 a* < α 가 되어 수량이 줄어든다.

완판 기록 (실제 수요 ≥ 판매량, 중도절단) 처리 — censored 옵션
  "strict" : 판매량으로 점수 계산. 실제 점수 이상이 되지 않으므로 보장이 그대로 성립하지만,
             과거에 적게 만들어 완판된 기록이 많으면 매우 보수적이 된다.
  "impute" : 모델을 믿고, 실제 점수가 [판매량의 점수, 1] 에 균등하게 있다고 보고 기대 위반 수로 계산.
             덜 보수적이지만 완판 기록 부분은 모델에 의존한다 (완판 기록이 적으면 strict 와 거의 같다).

필요한 최소 건수: n ≥ 1/α − 1 (α=0.1 → 9건, α=0.05 → 19건). 부족하면 보정할 수 없다.
주의: 보장은 보정 기록들과 비슷한 상황 '전체에 대한 평균' 이다 (개별 상황마다 보장되는 것은 아님).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LEVEL_GRID = np.concatenate([np.linspace(0.0005, 0.02, 40), np.linspace(0.021, 0.5, 480)])


def min_records(alpha: float) -> int:
    return math.ceil(1 / alpha - 1)


@dataclass
class LevelConformal:
    alpha: float
    censored: str = "strict"            # "strict" | "impute"
    allow_loosen: bool = False          # True 면 모델이 너무 보수적일 때 a* > α 도 허용
    level: float | None = None          # 보정된 분위수 수준 a*
    n: int = 0
    n_censored: int = 0

    def fit(self, lower_by_level: np.ndarray, sold: np.ndarray, sold_out: np.ndarray,
            levels: np.ndarray = LEVEL_GRID) -> "LevelConformal":
        """lower_by_level: (n, A) 각 기록에서 수준 levels[a] 의 안전 수량 L_i(a)."""
        L = np.asarray(lower_by_level, float)
        sold = np.asarray(sold, float)
        cens = np.asarray(sold_out, bool)
        self.n, self.n_censored = len(sold), int(cens.sum())
        if math.floor((self.n + 1) * self.alpha) < 1:
            raise ValueError(f"보정용 판매 기록이 부족합니다: {self.n}건 (α={self.alpha} 이면 최소 "
                             f"{min_records(self.alpha)}건 필요)")
        ok = L <= sold[:, None]                                    # (n, A), a 가 커질수록 False 로 바뀜
        idx = np.where(ok.any(1), ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
        s = np.where(idx >= 0, levels[np.maximum(idx, 0)], 0.0)    # 관측 판매량 기준 점수
        cand = levels[levels <= (1.0 if self.allow_loosen else self.alpha) + 1e-12]
        budget = (self.n + 1) * self.alpha - 1                     # 새 상황 1건 몫을 뺀 허용 위반 수
        if self.censored == "strict":
            k = math.floor((self.n + 1) * self.alpha)
            a = float(np.sort(s)[k - 1])
            self.level = min(a, cand.max())
        elif self.censored == "impute":
            best = cand.min()
            for a in cand:                                         # 기대 위반 수 ≤ budget 인 가장 큰 a
                viol = np.where(cens,
                                np.where(a <= s, 0.0, (a - s) / np.maximum(1 - s, 1e-9)),
                                (s < a).astype(float))
                if viol.sum() <= budget + 1e-9:
                    best = a
                else:
                    break
            self.level = float(best)
        else:
            raise ValueError("censored 는 'strict' 또는 'impute'")
        self.level = max(self.level, float(levels.min()))
        return self
