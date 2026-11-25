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
    # 판매 기록에 없는 종류 → 평균 굿즈 종류로 계산 (불확실성이 더 커서 수량이 더 보수적)
    feats = {f: None for f in NUMERIC_FEATURES}
    unknown = deterministic_recommend(model, feats | {"_price_ref": 12000}, "우산", [12000], CostSpec(unit_cost=4000))
    assert unknown["known_type"] is False and unknown["rows"][0]["safe_qty"] >= 0
    from salesopt.publish import member_params
    mu_u, sg_u = member_params(model, feats | {"_price_ref": 12000}, "우산", 12000)
    sg_known = np.mean([member_params(model, feats, t, 10000)[1].mean() for t in model["categories"]])
    assert sg_u.mean() > sg_known


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
        ("우산", CostSpec(5000, 0, 100000, 10, None), [12000, 15000, 18000], 0.1),   # 학습하지 않은 종류
    ]:
        ctx = sim.contexts(1, goods_type=gt if gt in model["categories"] else "키링")
        feats = {f: float(ctx[f].iloc[0]) for f in NUMERIC_FEATURES}
        feats["group_type_buzz_7d"] = None
        if gt not in model["categories"]:
            feats["_price_ref"] = 15000
        cases.append({"goods_type": gt, "features": feats, "prices": prices, "alpha": alpha,
                      "cost": {"unit_cost": cost.unit_cost, "salvage": cost.salvage, "fixed_cost": cost.fixed_cost,
                               "moq": cost.moq, "max_qty": cost.max_qty},
                      "expected": deterministic_recommend(model, feats, gt, prices, cost, alpha)})
    out = {"model": model, "cases": cases}
    try:                                        # 딥러닝(cVAE) 모델도 웹과 같은지 확인할 데이터
        import torch  # noqa: F401
        cpred = fit_predictor(sim.records(400), NUMERIC_FEATURES, kind="cvae", n_members=2, epochs=60, seed=3)
        cmodel = export_model(cpred, "real", conformal_level=0.07)
        ccases = []
        for gt, cost, prices in [("키링", CostSpec(4000, 500, 300000, 50, None), [9000, 12000, 15000]),
                                 ("응원봉", CostSpec(16000, 0, 0, 1, None), [45000])]:
            ctx = sim.contexts(1, goods_type=gt)
            feats = {f: float(ctx[f].iloc[0]) for f in NUMERIC_FEATURES}
            ccases.append({"goods_type": gt, "features": feats, "prices": prices, "alpha": 0.1,
                           "cost": {"unit_cost": cost.unit_cost, "salvage": cost.salvage,
                                    "fixed_cost": cost.fixed_cost, "moq": cost.moq, "max_qty": cost.max_qty},
                           "expected": deterministic_recommend(cmodel, feats, gt, prices, cost, 0.1)})
        out["cvae"] = {"model": cmodel, "cases": ccases}
    except ImportError:
        pass
    VECTORS.parent.mkdir(parents=True, exist_ok=True)
    VECTORS.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def test_real_models_trains_and_selects_cvae_when_enough_records(session, monkeypatch):
    """기록 100건 이상: 단순 모델과 딥러닝을 모두 학습하고, 검증 기록 점수로 '자동' 모델을 고른다."""
    pytest.importorskip("torch")
    from datetime import date

    import salesopt.features as feats_mod
    from salesopt.publish import deterministic_recommend, log_score, real_models
    from salesopt.storage import SalesRecord

    sim = Simulator(seed=21)
    recs = sim.records(140)
    recs["group_key"] = "g"
    recs["sale_start"] = date(2026, 11, 1)
    for i, r in recs.iterrows():
        session.add(SalesRecord(group_key="g", goods_type=r.goods_type, product_name=f"p{i}", price=float(r.price),
                                quantity_made=float(r.quantity_made), quantity_sold=float(r.quantity_sold),
                                sold_out=bool(r.sold_out), sale_start=date(2026, 11, 1), channel="", source="real"))
    session.flush()
    monkeypatch.setattr(feats_mod, "build_training_frame", lambda s, sales, gm: recs.copy())   # 특징은 시뮬레이션 값
    out = real_models(session, {})
    assert {"real", "real_lognormal", "real_cvae"} <= set(out)
    ln, cv = out["real_lognormal"], out["real_cvae"]
    assert ln["model_type"] == "lognormal" and cv["model_type"] == "cvae" and len(cv["z_nodes"]) == 64
    assert out["real"]["meta"]["selected"] in ("lognormal", "cvae")
    sel = "real_" + out["real"]["meta"]["selected"]
    assert out[sel]["meta"]["validation_log_score"] == max(ln["meta"]["validation_log_score"],
                                                           cv["meta"]["validation_log_score"])
    # 딥러닝 모델로도 추천이 계산되고, 학습하지 않은 종류는 지원하지 않는다
    f = {k: float(recs[k].iloc[0]) for k in NUMERIC_FEATURES}
    rec = deterministic_recommend(cv, f, "키링", [10000, 12000, 14000], default_cost("키링"))
    assert len(rec["rows"]) == 3 and all(r["demand_p10"] <= r["demand_p50"] <= r["demand_p90"] for r in rec["rows"])
    with pytest.raises(KeyError):
        deterministic_recommend(cv, f, "우산", [10000], default_cost("키링"))
    assert np.isfinite(log_score(cv, recs.head(10)))


def test_cvae_deterministic_matches_sampling():
    """웹이 쓰는 '대표점 64개' 계산이 신경망에서 직접 뽑은 샘플과 같은 분포를 내는지."""
    pytest.importorskip("torch")
    sim = Simulator(seed=22)
    pred = fit_predictor(sim.records(600), NUMERIC_FEATURES, kind="cvae", n_members=1, epochs=120)
    model = export_model(pred, "real")
    ctx = sim.contexts(1, goods_type="인형")
    feats = {f: float(ctx[f].iloc[0]) for f in NUMERIC_FEATURES}
    det = deterministic_recommend(model, feats, "인형", [30000], CostSpec(unit_cost=11000), alpha=0.1)["rows"][0]
    S = pred.sample_members(ctx, [30000], 60000, np.random.default_rng(0))[0, 0]
    assert det["demand_p50"] == pytest.approx(np.quantile(S, 0.5), rel=0.06, abs=3)
    assert det["safe_qty"] == pytest.approx(np.quantile(S, 0.1), rel=0.08, abs=3)
