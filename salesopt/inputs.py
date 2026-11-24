"""명령줄로 받는 비용·가격 입력.

한 값을 주면 모든 굿즈 종류에 적용하고, '종류=값' 으로 종류별로 줄 수 있다. 둘을 섞어도 된다.
  --cost 4000                      모든 종류 원가 4,000원
  --cost 키링=4000,인형=11000        종류별
  --cost 5000,인형=11000             기본 5,000원, 인형만 11,000원
  --price 키링=9000~16000            검토할 가격 범위 (종류=하한~상한). 하나만 주면 그 가격으로 고정
                                    비우면 판매 기록에 있던 가격 범위를 쓴다
금액에는 쉼표 대신 _ 나 '원' 을 써도 된다 (4_000, 4000원). 종류 구분 쉼표와 헷갈리지 않게 천 단위 쉼표는 쓰지 않는다.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import numpy as np

from salesopt.optimize import CostSpec


def _num(s: str) -> float:
    t = s.strip().replace("_", "").removesuffix("원").strip()
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"숫자가 아닙니다: '{s}'") from None


def _range(s: str) -> tuple[float, float]:
    parts = re.split(r"\s*[~]\s*|\s+-\s+|(?<=\d)-(?=\d)", s.strip())
    if len(parts) == 1:
        v = _num(parts[0])
        return v, v
    if len(parts) != 2:
        raise ValueError(f"가격 범위는 '하한~상한' 형식입니다: '{s}'")
    lo, hi = _num(parts[0]), _num(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


@dataclass
class PerType:
    """'기본값' + '종류별 값'."""
    default: object = None
    by_type: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str | None, cast=_num) -> "PerType":
        out = cls()
        if not text:
            return out
        for item in (x for x in text.split(",") if x.strip()):
            if "=" in item:
                k, v = item.split("=", 1)
                out.by_type[k.strip()] = cast(v)
            else:
                if out.default is not None:
                    raise ValueError(f"기본값이 두 번 들어왔습니다: '{text}' (종류별 값은 '종류=값')")
                out.default = cast(item)
        return out

    def get(self, goods_type: str, fallback=None):
        return self.by_type.get(goods_type, self.default if self.default is not None else fallback)

    def unknown_types(self, known) -> list[str]:
        return sorted(set(self.by_type) - set(known))


@dataclass
class CostInputs:
    cost: PerType
    salvage: PerType
    fixed_cost: PerType
    moq: PerType
    max_qty: PerType
    price: PerType
    price_step: float = 500.0
    interactive: bool = False         # 원가가 빠진 종류는 직접 물어본다 (터미널에서 실행할 때)

    @classmethod
    def from_args(cls, a, interactive: bool | None = None) -> "CostInputs":
        return cls(
            cost=PerType.parse(a.cost), salvage=PerType.parse(a.salvage), fixed_cost=PerType.parse(a.fixed_cost),
            moq=PerType.parse(a.moq), max_qty=PerType.parse(a.max_qty), price=PerType.parse(a.price, _range),
            price_step=_num(str(a.price_step)),
            interactive=sys.stdin.isatty() if interactive is None else interactive,
        )

    def all_types(self) -> set[str]:
        return set().union(*(p.by_type for p in (self.cost, self.salvage, self.fixed_cost, self.moq,
                                                  self.max_qty, self.price)))

    def _ask(self, goods_type: str) -> float:
        while True:
            raw = input(f"'{goods_type}' 개당 원가(원)를 입력하세요: ").strip()
            try:
                v = _num(raw)
                self.cost.by_type[goods_type] = v
                return v
            except ValueError as e:
                print(f"  {e}")

    def spec(self, goods_type: str) -> CostSpec:
        unit = self.cost.get(goods_type)
        if unit is None:
            if not self.interactive:
                raise SystemExit(f"'{goods_type}' 의 개당 원가가 없습니다: --cost {goods_type}=원가 (또는 --cost 원가)")
            unit = self._ask(goods_type)
        salvage = self.salvage.get(goods_type, 0.0)
        if salvage >= unit:
            raise SystemExit(f"'{goods_type}' 처분가({salvage:,.0f}원)가 원가({unit:,.0f}원) 이상입니다")
        mq = self.max_qty.get(goods_type)
        return CostSpec(unit_cost=unit, salvage=salvage, fixed_cost=self.fixed_cost.get(goods_type, 0.0),
                        moq=int(self.moq.get(goods_type, 1)), max_qty=int(mq) if mq is not None else None)

    def grid(self, goods_type: str, support: tuple[float, float] | None) -> tuple[np.ndarray, list[str]]:
        """검토할 가격 격자. 입력이 없으면 학습 데이터 가격 범위."""
        notes = []
        rng = self.price.get(goods_type)
        if rng is None:
            if support is None:
                raise SystemExit(f"'{goods_type}' 가격 범위를 정할 수 없습니다: --price {goods_type}=하한~상한")
            lo, hi = support
            notes.append(f"가격 범위를 입력하지 않아 판매 기록의 가격 범위 {lo:,.0f}~{hi:,.0f}원을 검토했습니다")
        else:
            lo, hi = rng
            if support and (lo < support[0] or hi > support[1]):
                notes.append(f"가격 {lo:,.0f}~{hi:,.0f}원 중 판매 기록 범위({support[0]:,.0f}~{support[1]:,.0f}원) "
                             "밖은 외삽이라 덜 믿을 만합니다")
        if lo == hi:
            return np.array([lo]), notes
        step = self.price_step
        grid = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
        if len(grid) == 0 or grid[-1] < hi:
            grid = np.append(grid, hi)
        return grid, notes


def add_cost_args(p, required_cost: bool) -> None:
    g = p.add_argument_group("비용·가격 입력 (한 값 = 모든 종류, '종류=값' = 종류별, 쉼표로 여러 개)")
    g.add_argument("--cost", required=required_cost, help="개당 제작 원가. 예) 4000  또는  키링=4000,인형=11000")
    g.add_argument("--salvage", default="", help="못 판 재고 개당 처분가 (기본 0)")
    g.add_argument("--fixed-cost", default="", help="1회 생산 고정비: 디자인·금형 등 (기본 0)")
    g.add_argument("--moq", default="", help="최소 생산 수량 (기본 1)")
    g.add_argument("--max-qty", default="", help="생산 가능 최대 수량 (기본 제한 없음)")
    g.add_argument("--price", default="", help="검토할 판매가. 예) 키링=9000~16000 (하나만 주면 고정 가격). "
                                                "비우면 판매 기록의 가격 범위")
    g.add_argument("--price-step", default="500", help="가격 격자 간격 (기본 500원)")
