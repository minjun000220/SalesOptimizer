import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import FIX
from salesopt.models import fit_predictor
from salesopt.optimize import CostSpec, recommend
from salesopt.publish import deterministic_recommend, export_model, publish
from salesopt.simulate import NUMERIC_FEATURES, Simulator, default_cost
from salesopt.sources.idol import ingest
from salesopt.storage import PublishedContext, PublishedModel

VECTORS = Path(__file__).resolve().parent.parent / "web" / "lib" / "test-vectors.json"


@pytest.fixture(scope="module")
def trained():
    sim = Simulator(seed=7)
    pred = fit_predictor(sim.records(500), NUMERIC_FEATURES, n_members=6, seed=7)
    return sim, pred, export_model(pred, "demo")


def test_deterministic_matches_monte_carlo(trained):
    sim, pred, model = trained
    for i, gt in enumerate(["키링", "응원봉", "인형"]):
        ctx = sim.contexts(1, goods_type=gt)
        feats = {f: float(ctx[f].iloc[0]) for f in NUMERIC_FEATURES}
        cost = default_cost(gt)
        ref = {"키링": 12000, "응원봉": 45000, "인형": 32000}[gt]
        grid = ref * np.array([0.8, 1.0, 1.2])
        det = deterministic_recommend(model, feats, gt, grid, cost, alpha=0.1)
        mc = recommend(pred, ctx, grid, cost, alpha=0.1, m=40000, seed=i).table
        for r, (_, m) in zip(det["rows"], mc.iterrows()):
            assert r["price"] == m.price
            assert r["safe_qty"] == pytest.approx(m.safe_qty, rel=0.03, abs=2)
            assert r["expected_profit"] == pytest.approx(m.expected_profit, rel=0.03, abs=2000)
            assert r["demand_p50"] == pytest.approx(m.demand_p50, rel=0.03, abs=2)
            if r["sellout_prob"] is not None:
                assert r["sellout_prob"] == pytest.approx(m.sellout_prob, abs=0.01)


def test_missing_features_use_training_mean(trained):
    _, _, model = trained
    det_all_none = deterministic_recommend(model, {f: None for f in NUMERIC_FEATURES}, "키링", [12000],
                                           CostSpec(unit_cost=4000), alpha=0.1)
    assert det_all_none["rows"][0]["safe_qty"] > 0
    with pytest.raises(KeyError):
        deterministic_recommend(model, {}, "우산", [12000], CostSpec(unit_cost=4000))


def test_publish_writes_model_and_contexts(session):
    ingest(session, str(FIX / "idol"))
    out = publish(session, {})
    assert out["demo"] == "created" and "부족" in out["real"]
    m = session.get(PublishedModel, "demo")
    assert len(m.payload["members"]) == 10 and m.payload["numeric"] == NUMERIC_FEATURES
    ctxs = session.query(PublishedContext).all()
    assert {c.group_key for c in ctxs} == {"illit"} and len(ctxs) >= 6
    c = next(c for c in ctxs if c.goods_type == "키링")
    assert c.features["favor"] == 58.1 and c.features["hot_index"] is None and c.group_name == "아일릿"


def test_write_vectors_for_typescript(trained):
    """web/lib/optimize.ts 가 같은 결과를 내는지 확인하는 데이터 (web 쪽 테스트가 읽는다)."""
    sim, _, model = trained
    cases = []
    for gt, cost, prices, alpha in [
        ("키링", CostSpec(4000, 500, 300000, 50, None), [9000, 11000, 13000, 15000], 0.1),
        ("응원봉", CostSpec(16000, 0, 0, 1, 900), [40000, 50000], 0.05),
        ("인형", CostSpec(11000, 0, 0, 1, None), [30000], 0.2),
    ]:
        ctx = sim.contexts(1, goods_type=gt)
        feats = {f: float(ctx[f].iloc[0]) for f in NUMERIC_FEATURES}
        feats["group_type_buzz_7d"] = None
        cases.append({"goods_type": gt, "features": feats, "prices": prices, "alpha": alpha,
                      "cost": {"unit_cost": cost.unit_cost, "salvage": cost.salvage, "fixed_cost": cost.fixed_cost,
                               "moq": cost.moq, "max_qty": cost.max_qty},
                      "expected": deterministic_recommend(model, feats, gt, prices, cost, alpha)})
    VECTORS.parent.mkdir(parents=True, exist_ok=True)
    VECTORS.write_text(json.dumps({"model": model, "cases": cases}, ensure_ascii=False), encoding="utf-8")
