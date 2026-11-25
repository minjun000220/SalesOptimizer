import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, text

from conftest import FIX
from salesopt.features import GroupInfo, build_context, build_training_frame, select_candidates
from salesopt.schema import load_sales_csv, write_template
from salesopt.sources.goods import group_type_buzz, hot_scores, kream_proxy_records
from salesopt.sources.idol import ingest
from salesopt.storage import IdolSnapshot
from salesopt.tagger import GroupTagger


def test_ingest_idol_and_history(session):
    stats = ingest(session, str(FIX / "idol"))
    assert stats["groups"] == 1 and stats["added"] == 3
    rows = session.scalars(select(IdolSnapshot).order_by(IdolSnapshot.date)).all()
    assert [r.date for r in rows] == [date(2026, 11, 14), date(2026, 11, 16), date(2026, 11, 23)]
    assert [r.complete for r in rows] == [False, False, True]
    assert rows[-1].controversy == 0.21
    assert rows[-1].member_hhi == pytest.approx(0.6 ** 2 + 0.4 ** 2)
    again = ingest(session, str(FIX / "idol"))                      # 다시 실행해도 중복 없이
    assert again["added"] == 0
    assert len(session.scalars(select(IdolSnapshot)).all()) == 3


def test_tagger():
    t = GroupTagger.from_idol_configs(str(FIX / "idol_config"))
    assert t.tag("BTS 공식 응원봉 Ver. 4") == "bts"
    assert t.tag("아일릿 ILLIT 아크릴 키링") == "illit"
    assert t.tag("BTS x 아일릿 콜라보") is None                      # 두 그룹 → 판단 안 함
    assert t.tag("abtsx 키링") is None                               # 영문 단어 경계
    assert t.tag("산리오 쿠로미 인형") is None


def _seed_goods(session, today: date):
    bd = {"hot_index": 72, "signals": {"buzz_momentum": {"z": 1.5, "value": 0.4},
                                       "sold_out_ratio": {"z": 0.5, "value": 0.35}}}
    bd2 = {"hot_index": 40, "signals": {"buzz_momentum": {"z": -0.5, "value": -0.1}}}
    session.execute(text("INSERT INTO hot_score (goods_type, as_of, score, rank, breakdown) VALUES "
                         "(:a, :d, 0.6, 1, :b), (:a2, :d, -0.1, 2, :b2)"),
                    {"a": "키링", "a2": "인형", "d": today - timedelta(days=1), "b": json.dumps(bd, ensure_ascii=False),
                     "b2": json.dumps(bd2, ensure_ascii=False)})
    for i in range(5):
        session.execute(text("INSERT INTO mention (source_id, external_id, goods_type, text, url, observed_at, engagement) "
                             "VALUES ('dc_illit', :e, '인형', 't', 'u', :t, 0)"),
                        {"e": f"m{i}", "t": datetime.combine(today - timedelta(days=2), datetime.min.time())})
    session.execute(text("INSERT INTO mention (source_id, external_id, goods_type, text, url, observed_at, engagement) "
                         "VALUES ('dc_illit', 'late', '인형', 't', 'u', :t, 0)"),
                    {"t": datetime.combine(today, datetime.min.time()) + timedelta(hours=1)})   # 당일 → 제외
    session.flush()


def test_goods_sources(session):
    today = date(2026, 11, 24)
    _seed_goods(session, today)
    hot = hot_scores(session, today)
    assert hot.loc["키링", "hot_index"] == 72 and hot.loc["키링", "buzz_z"] == 1.5
    assert pd.isna(hot.loc["인형", "sold_out_ratio"])
    buzz = group_type_buzz(session, "illit", today)
    assert buzz.to_dict() == {"인형": 5.0}


def test_build_context_and_candidates(session):
    ingest(session, str(FIX / "idol"))
    _seed_goods(session, date(2026, 11, 24))
    g = GroupInfo("illit", "illit", [])
    ctx = build_context(session, g, "키링", date(2026, 11, 24)).iloc[0]
    assert ctx.favor == 58.1 and ctx.hot_index == 72
    assert ctx.favor_change_7d == pytest.approx(58.1 - 56.0)          # 7일 이상 전 가장 가까운 값 (11-16)
    assert ctx.log_mentions_per_day == pytest.approx(np.log(607.61))
    assert ctx.group_type_buzz_7d == 0.0
    ctx2 = build_context(session, GroupInfo("illit", None, []), "인형", date(2026, 11, 24)).iloc[0]
    assert np.isnan(ctx2.group_type_buzz_7d)                           # 갤러리 없음 → NaN
    assert set(select_candidates(session, g, date(2026, 11, 24), k=2)) == {"키링", "인형"}
    # 판매 시작일 이전 정보만 쓰는지 (누수 방지): 11-15 판매에는 11-14 스냅샷
    sales = pd.DataFrame([{"group_key": "illit", "goods_type": "키링", "sale_start": date(2026, 11, 15)}])
    tr = build_training_frame(session, sales, {"illit": g})
    assert tr.favor.iloc[0] == 55.0 and tr.idol_snapshot_date.iloc[0] == date(2026, 11, 14)


def test_kream_proxy(session):
    session.execute(text("INSERT INTO listing (id, source_id, external_id, goods_type, title_raw, url, seller_type) "
                         "VALUES (1, 'kream', '1', '응원봉', 'BTS 공식 응원봉 Ver. 4', 'u', 'resale'), "
                         "(2, 'kream', '2', '응원봉', '무명 응원봉', 'u', 'resale')"))
    start = date(2026, 11, 1)
    for lid in (1, 2):
        for k in range(8):
            session.execute(text("INSERT INTO listing_snapshot (listing_id, observed_at, price, sales_count) "
                                 "VALUES (:l, :t, :p, :c)"),
                            {"l": lid, "t": datetime.combine(start + timedelta(days=k), datetime.min.time()),
                             "p": 50000 + 1000 * k, "c": 7000 + 10 * k})
    tagger = GroupTagger({"bts": ["BTS", "방탄소년단"]})
    out = kream_proxy_records(session, tagger, start, start + timedelta(days=7), window_days=7)
    assert set(out.group_key) == {"bts"}                              # 그룹을 못 찾은 상품은 제외
    # 1~6일 증가분(6×10)이 첫 7일 구간. 7일째 증가분은 다음 구간인데 하루뿐이라 제외
    assert out.quantity_sold.tolist() == [60] and (out.source == "proxy").all()


def test_sales_csv(tmp_path):
    p = write_template(tmp_path / "t.csv")
    df = load_sales_csv(p, lambda g: {"아일릿": "illit"}[g])
    assert df.group_key.tolist() == ["illit", "illit"]
    assert df.sold_out.tolist() == [True, False]
    bad = pd.read_csv(p, encoding="utf-8-sig")
    bad.loc[0, "quantity_sold"] = 9999
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        load_sales_csv(tmp_path / "bad.csv")


def test_database_url_normalized(monkeypatch):
    from salesopt.storage import database_url, normalize_url
    assert normalize_url("postgresql://u:p@h:5432/d") == "postgresql+psycopg://u:p@h:5432/d"
    assert normalize_url("postgres://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert normalize_url("postgresql+psycopg://x") == "postgresql+psycopg://x"
    monkeypatch.setenv("DATABASE_URL", "")
    assert database_url().startswith("sqlite:///")
