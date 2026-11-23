"""논문 Algorithm 1·2 의 '안전 완판' 변형.

원래 논문 (기대이익 최대):
    q̂(x, p) = 생성 샘플의 ρ(p) 분위수,  ρ(p) = (p − c) / (p − s)
    p̂(x)    = argmax_p  π̂(x, p, q̂(x, p))   (가격 격자 위에서)

이 프로그램 (확률 제약: 완판 확률 ≥ 1 − α):
    max_{p,q} E[Π]  s.t.  P(D ≥ q | x, p) ≥ 1 − α
    ⇒ q*(p) = F⁻¹( min(α, ρ(p)) )

모델 불확실성까지 막기 위해 생성기 B개(앙상블)의 분위수 중 하위 값(conservative 백분위)을 쓰고,
실제 판매 기록이 있으면 conformal 보정(conformal.py)으로 분위수 수준 α 를 a* ≤ α 로 낮춘다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostSpec:
    unit_cost: float            # 개당 제작 원가 c
    salvage: float = 0.0        # 남은 재고 개당 처분가 s (못 팔면 0)
    fixed_cost: float = 0.0     # 1회 생산 고정비 (디자인·금형 등). 만들지 않으면 0
    moq: int = 1                # 최소 생산 수량. 안전 수량이 이보다 작으면 '만들지 않음'
    max_qty: int | None = None  # 생산 가능한 최대 수량


def profit(d: np.ndarray, price: float, q: float, cost: CostSpec) -> np.ndarray:
    """Π(d, p, q) = p·min(q, d) + s·(q − d)⁺ − c·q − F·1[q>0]"""
    d = np.asarray(d, dtype=float)
    if q <= 0:
        return np.zeros_like(d)
    sold = np.minimum(q, d)
    return price * sold + cost.salvage * (q - sold) - cost.unit_cost * q - cost.fixed_cost


def critical_ratio(price: float, cost: CostSpec) -> float:
    return (price - cost.unit_cost) / (price - cost.salvage)


def safe_quantity(member_samples: np.ndarray, level: float, conservative: float) -> float:
    """member_samples: (B, m). 멤버별 level 분위수 → 그중 conservative 백분위 (B=1 이면 그 값)."""
    qs = np.quantile(member_samples, level, axis=1)
    return float(qs[0] if len(qs) == 1 else np.quantile(qs, conservative))


@dataclass
class Recommendation:
    table: pd.DataFrame                 # 가격별 결과
    best: dict | None                   # 추천 (없으면 None = 안전하게 이익이 나는 가격이 없음)
    alpha: float
    notes: list[str] = field(default_factory=list)


def recommend(predictor, context: pd.DataFrame, price_grid, cost: CostSpec, alpha: float = 0.1,
              conservative: float = 0.1, m: int = 4000, conformal=None, seed: int = 0) -> Recommendation:
    """한 (그룹, 굿즈 종류) 상황 context(1행)에 대해 가격 격자별 안전 수량·완판 확률·기대이익을 계산한다.

    predictor.sample_members(context, prices, m, rng) → (B, J, m) 수요 샘플 (논문 식 (8))
    """
    if len(context) != 1:
        raise ValueError("context 는 1행이어야 합니다")
    prices = np.asarray(sorted(set(float(p) for p in price_grid)))
    rng = np.random.default_rng(seed)
    S = predictor.sample_members(context, prices, m, rng)           # (B, J, m)
    rows, notes = [], []
    for j, p in enumerate(prices):
        if p <= cost.unit_cost:
            continue
        Sj = S[:, j, :]
        pooled = Sj.reshape(-1)
        rho = critical_ratio(p, cost)
        level = min(conformal.level if conformal is not None else alpha, rho)   # 보정된 수준 a* ≤ α
        q = np.floor(safe_quantity(Sj, level, conservative))
        if cost.max_qty is not None:
            q = min(q, cost.max_qty)
        if q < cost.moq:
            q = 0.0
        q_nv = float(np.floor(np.quantile(pooled, rho)))              # 논문 원래 해 (참고용)
        member_ok = (Sj >= q).mean(axis=1) >= 1 - alpha if q > 0 else np.ones(len(Sj), bool)
        rows.append({
            "price": p,
            "safe_qty": int(q),
            "sellout_prob": float((pooled >= q).mean()) if q > 0 else np.nan,
            "expected_profit": float(profit(pooled, p, q, cost).mean()),
            "expected_revenue": float((p * np.minimum(q, pooled)).mean()) if q > 0 else 0.0,
            "expected_leftover": float(np.maximum(q - pooled, 0).mean()) if q > 0 else 0.0,
            "model_agreement": float(member_ok.mean()),
            "demand_p10": float(np.quantile(pooled, 0.1)),
            "demand_p50": float(np.quantile(pooled, 0.5)),
            "demand_p90": float(np.quantile(pooled, 0.9)),
            "newsvendor_qty": int(q_nv),
            "newsvendor_sellout_prob": float((pooled >= q_nv).mean()),
            "newsvendor_profit": float(profit(pooled, p, q_nv, cost).mean()) if q_nv >= cost.moq else 0.0,
        })
    table = pd.DataFrame(rows)
    best = None
    if not table.empty:
        ok = table[(table.safe_qty > 0) & (table.expected_profit > 0)]
        if not ok.empty:
            best = ok.loc[ok.expected_profit.idxmax()].to_dict()
            best["safe_qty"] = int(best["safe_qty"])
            best["newsvendor_qty"] = int(best["newsvendor_qty"])
            if len(table) > 2 and best["price"] in (table.price.min(), table.price.max()):
                edge = "최고가" if best["price"] == table.price.max() else "최저가"
                notes.append(f"추천 가격이 검토 범위의 {edge}입니다 → 범위를 넓히면 더 나은 가격이 있을 수 있습니다")
        else:
            notes.append("완판 확률을 지키면서 이익이 나는 가격이 없습니다 (만들지 않는 것을 권장)")
    if conformal is None:
        notes.append("실제 판매 기록으로 보정(conformal)하지 않은 결과입니다")
    return Recommendation(table, best, alpha, notes)
