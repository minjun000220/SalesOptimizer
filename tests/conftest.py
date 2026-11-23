import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = Path(__file__).parent / "fixtures"

# Goods(goods-trend) 의 테이블 중 SalesOptimizer 가 읽는 열만 (storage/models.py 와 같은 이름)
GOODS_DDL = [
    "CREATE TABLE hot_score (id INTEGER PRIMARY KEY, goods_type TEXT, as_of DATE, score FLOAT, rank INTEGER, breakdown JSON)",
    "CREATE TABLE mention (id INTEGER PRIMARY KEY, source_id TEXT, external_id TEXT, goods_type TEXT, text TEXT, "
    "url TEXT, observed_at DATETIME, engagement INTEGER)",
    "CREATE TABLE listing (id INTEGER PRIMARY KEY, source_id TEXT, external_id TEXT, goods_type TEXT, title_raw TEXT, "
    "url TEXT, seller_type TEXT)",
    "CREATE TABLE listing_snapshot (id INTEGER PRIMARY KEY, listing_id INTEGER, observed_at DATETIME, price INTEGER, "
    "rank INTEGER, rank_list TEXT, stock_status TEXT, review_count INTEGER, sales_count INTEGER)",
]


@pytest.fixture
def session():
    from salesopt.storage import Base
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with eng.begin() as c:
        for ddl in GOODS_DDL:
            c.execute(text(ddl))
    with Session(eng) as s:
        yield s
