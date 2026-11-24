from types import SimpleNamespace

import numpy as np
import pytest

from salesopt.inputs import CostInputs, PerType


def args(**kw):
    base = dict(cost="", salvage="", fixed_cost="", moq="", max_qty="", price="", price_step="500")
    return SimpleNamespace(**(base | kw))


def test_per_type_parse():
    p = PerType.parse("5000,인형=11000, 키링 = 4_000원")
    assert p.default == 5000 and p.by_type == {"인형": 11000, "키링": 4000}
    assert p.get("인형") == 11000 and p.get("응원봉") == 5000
    assert PerType.parse("").get("키링", 7) == 7
    with pytest.raises(ValueError):
        PerType.parse("4000,5000")
    with pytest.raises(ValueError):
        PerType.parse("키링=사천원")


def test_cost_spec_and_defaults():
    c = CostInputs.from_args(args(cost="키링=4000", salvage="500", fixed_cost="키링=300000", moq="50",
                                  max_qty="키링=2000"), interactive=False)
    s = c.spec("키링")
    assert (s.unit_cost, s.salvage, s.fixed_cost, s.moq, s.max_qty) == (4000, 500, 300000, 50, 2000)
    with pytest.raises(SystemExit):
        c.spec("인형")                                   # 원가 없음 + 비대화형 → 안내 후 종료
    with pytest.raises(SystemExit):
        CostInputs.from_args(args(cost="1000", salvage="1000"), interactive=False).spec("키링")


def test_cost_asked_interactively(monkeypatch):
    answers = iter(["모름", "11000"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    c = CostInputs.from_args(args(), interactive=True)
    assert c.spec("인형").unit_cost == 11000
    assert c.spec("인형").unit_cost == 11000             # 한 번 물어본 값은 기억


def test_price_grid():
    c = CostInputs.from_args(args(cost="4000", price="키링=9000~10000,인형=30000", price_step="400"),
                             interactive=False)
    g, notes = c.grid("키링", (9500, 12000))
    assert g.tolist() == [9200, 9600, 10000] and notes            # 판매 기록 범위 밖 → 외삽 경고
    assert c.grid("인형", None)[0].tolist() == [30000]            # 한 값 = 고정 가격
    g2, notes2 = c.grid("응원봉", (40000, 41000))                 # 입력 없음 → 판매 기록 범위
    assert g2[0] == 40000 and g2[-1] == 41000 and notes2
    with pytest.raises(SystemExit):
        c.grid("응원봉", None)
    assert CostInputs.from_args(args(cost="1", price="9000-10000"), interactive=False).grid("x", None)[0][0] == 9000
    assert np.all(np.diff(g) > 0)
