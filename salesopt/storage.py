"""SalesOptimizer 전용 테이블. Goods 와 같은 Supabase DB 를 쓰되 이름 앞에 so_ 를 붙여 섞이지 않게 한다.

so_idol_snapshot : IdolSentiments 결과를 날짜별로 영구 보관 (IdolSentiments 는 90일 추이만 남기므로)
so_sales_record  : 실제 판매 기록 (나중에 추가). 모델 학습·conformal 보정에 쓴다
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import UTC, date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class IdolSnapshot(Base):
    __tablename__ = "so_idol_snapshot"
    __table_args__ = (UniqueConstraint("group_key", "date"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    group_key: Mapped[str] = mapped_column(String(80), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    name: Mapped[str] = mapped_column(String(120))
    favor: Mapped[float | None] = mapped_column(Float)
    interest: Mapped[float | None] = mapped_column(Float)
    controversy: Mapped[float | None] = mapped_column(Float)
    net: Mapped[float | None] = mapped_column(Float)
    mentions_per_day: Mapped[float | None] = mapped_column(Float)
    n_mentions: Mapped[int | None] = mapped_column(Integer)
    member_hhi: Mapped[float | None] = mapped_column(Float)
    members: Mapped[dict] = mapped_column(JSON, default=dict)       # 멤버 이름 → {favor, interest, share ...}
    sentiment_model: Mapped[str | None] = mapped_column(String(40))
    complete: Mapped[bool] = mapped_column(Boolean, default=True)   # False = 추이(history)에서 복원한 일부 값
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None))


class SalesRecord(Base):
    __tablename__ = "so_sales_record"
    __table_args__ = (UniqueConstraint("group_key", "goods_type", "product_name", "sale_start", "channel"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    group_key: Mapped[str] = mapped_column(String(80), index=True)
    goods_type: Mapped[str] = mapped_column(String(50), index=True)
    product_name: Mapped[str] = mapped_column(String(300), default="")
    price: Mapped[float] = mapped_column(Float)
    unit_cost: Mapped[float | None] = mapped_column(Float)
    quantity_made: Mapped[float | None] = mapped_column(Float)
    quantity_sold: Mapped[float] = mapped_column(Float)
    sold_out: Mapped[bool] = mapped_column(Boolean)
    sale_start: Mapped[date] = mapped_column(Date)
    sale_end: Mapped[date | None] = mapped_column(Date)
    sold_out_at: Mapped[datetime | None] = mapped_column(DateTime)
    channel: Mapped[str] = mapped_column(String(80), default="")
    source: Mapped[str] = mapped_column(String(20), default="real")   # real | proxy


def database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///data/salesopt.db")


_engine = None
_Session = None


def init_db(url: str | None = None):
    global _engine, _Session
    url = url or database_url()
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        os.makedirs(os.path.dirname(url.removeprefix("sqlite:///")) or ".", exist_ok=True)
    _engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


@contextmanager
def session_scope() -> Session:
    if _Session is None:
        init_db()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
