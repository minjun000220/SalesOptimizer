"""Goods(goods-trend) 테이블 읽기. 같은 Supabase DB 의 hot_score, mention, listing, listing_snapshot 을 쓴다.

  - 굿즈 종류 점수   : hot_score.breakdown 의 핫지수와 신호별 z 값
  - 그룹 × 종류 언급 : Goods 가 크롤링하는 아이돌 갤러리(source_id = 'dc_<갤러리 id>') 의 종류별 언급 수
  - KREAM 대리 수요  : 상품별 누적 거래 수의 증가량 (리셀 거래라 1차 판매와 다르다 → source='proxy')
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import logging

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)
_warned: set[str] = set()


def _has(session: Session, table: str) -> bool:
    """Goods 테이블이 이 DB 에 있는지 (로컬에서 Goods 없이 실행하는 경우 → 빈 결과 + 경고 1회)."""
    ok = inspect(session.connection()).has_table(table)
    if not ok and table not in _warned:
        _warned.add(table)
        log.warning("Goods 테이블 '%s' 이 DB 에 없습니다 → 굿즈 특징 없이 진행 (DATABASE_URL 을 확인하세요)", table)
    return ok


def _bd(v) -> dict:
    if isinstance(v, dict):
        return v
    return json.loads(v) if v else {}


def hot_scores(session: Session, as_of: date) -> pd.DataFrame:
    """as_of 이전(포함) 가장 최근 계산일의 종류별 점수 → index=goods_type."""
    if not _has(session, "hot_score"):
        return pd.DataFrame()
    d = session.execute(text("SELECT MAX(as_of) FROM hot_score WHERE as_of <= :d"), {"d": as_of}).scalar()
    if d is None:
        return pd.DataFrame()
    rows = session.execute(text("SELECT goods_type, score, rank, breakdown FROM hot_score WHERE as_of = :d"),
                           {"d": d}).all()
    out = []
    for gt, score, rank, bd in rows:
        bd = _bd(bd)
        sig = bd.get("signals", {})
        z = lambda k: (sig.get(k) or {}).get("z")                       # noqa: E731
        val = lambda k: (sig.get(k) or {}).get("value")                 # noqa: E731
        out.append({"goods_type": gt, "score": score, "rank": rank, "hot_index": bd.get("hot_index"),
                    "buzz_z": z("buzz_momentum"), "sold_out_ratio": val("sold_out_ratio"),
                    "best_share_z": z("best_share_momentum"), "resale_velocity_z": z("resale_trade_velocity"),
                    "scored_on": d})
    return pd.DataFrame(out).set_index("goods_type")


def group_type_buzz(session: Session, gallery_id: str | None, as_of: date, days: int = 7) -> pd.Series | None:
    """그룹 갤러리에서 최근 days 일 동안 굿즈 종류별 언급 수 (as_of 당일 제외, 미래 정보 누수 방지)."""
    if not gallery_id or not _has(session, "mention"):
        return None                                   # 모름 (0 이 아님)
    end = datetime.combine(as_of, time.min)
    rows = session.execute(text(
        "SELECT goods_type, COUNT(*) FROM mention WHERE source_id = :s AND goods_type IS NOT NULL "
        "AND observed_at >= :a AND observed_at < :b GROUP BY goods_type"),
        {"s": f"dc_{gallery_id}", "a": end - timedelta(days=days), "b": end}).all()
    return pd.Series({gt: float(n) for gt, n in rows}, dtype=float)


def kream_proxy_records(session: Session, tagger, start: date, end: date, window_days: int = 7) -> pd.DataFrame:
    """KREAM 상품 제목에서 그룹을 찾고, window_days 단위로 '거래 수 증가량'을 대리 판매량으로 만든다.
    가격 = 그 기간 시세 중앙값. 생산 수량·완판 정보가 없으므로 sold_out=False, quantity_made=NaN."""
    if not (_has(session, "listing") and _has(session, "listing_snapshot")):
        return pd.DataFrame()
    rows = session.execute(text(
        "SELECT l.id, l.title_raw, l.goods_type, s.observed_at, s.price, s.sales_count "
        "FROM listing l JOIN listing_snapshot s ON s.listing_id = l.id "
        "WHERE l.source_id = 'kream' AND l.goods_type IS NOT NULL AND s.sales_count IS NOT NULL "
        "AND s.observed_at >= :a AND s.observed_at < :b"),
        {"a": datetime.combine(start, time.min), "b": datetime.combine(end + timedelta(days=1), time.min)}).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["listing_id", "title", "goods_type", "observed_at", "price", "sales_count"])
    df["group_key"] = df.title.map(tagger.tag)
    df = df[df.group_key.notna()].copy()
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df.observed_at).dt.normalize()
    daily = df.groupby(["listing_id", "day"]).agg(sales=("sales_count", "max"), price=("price", "median"),
                                                  title=("title", "first"), goods_type=("goods_type", "first"),
                                                  group_key=("group_key", "first")).reset_index()
    daily = daily.sort_values(["listing_id", "day"])
    daily["inc"] = daily.groupby("listing_id").sales.diff().clip(lower=0)
    daily = daily.dropna(subset=["inc"])
    daily["window"] = ((daily.day - pd.Timestamp(start)).dt.days // window_days)
    out = daily.groupby(["listing_id", "window"]).agg(
        group_key=("group_key", "first"), goods_type=("goods_type", "first"), product_name=("title", "first"),
        price=("price", "median"), quantity_sold=("inc", "sum"), days=("day", "nunique"),
        sale_start=("day", "min")).reset_index()
    out = out[(out.days >= max(2, window_days // 2)) & out.price.notna() & (out.price > 0)]
    out["sale_start"] = out.sale_start.dt.date
    out["quantity_made"] = np.nan
    out["sold_out"] = False
    out["channel"] = "kream"
    out["source"] = "proxy"
    return out.drop(columns=["listing_id", "window", "days"]).reset_index(drop=True)


MIN_OVERLAP_TYPES = 3          # 판매가·리셀가가 둘 다 있는 종류가 이만큼 있어야 비율을 믿는다


def goods_price_stats(session: Session, as_of: date, days: int = 90, min_n: int = 5) -> pd.DataFrame:
    """굿즈 종류별 가격 분포 (최근 days 일): 보통 가격(중앙값)과 10~90% 범위. index=goods_type.
    판매 기록이 없는 종류를 계산할 때 기준 가격·검토할 가격 범위로 쓴다.

    source 열
      retail : 판매처 가격 (애니메이트·아이돌 굿즈몰 등)
      resale : 판매가 자료가 없어 KREAM 리셀 시세로 대신한 값
               - 판매가·리셀가가 둘 다 있는 종류가 3개 이상이면, 그 종류들의 '판매가 ÷ 리셀가' 중앙값을
                 비율로 곱해 판매가를 추정 (resale_ratio 열)
               - 아니면 비율을 모르므로 범위 하한을 리셀가 10% 지점의 절반까지 넓힌다 (정가는 리셀가보다 낮음)
    """
    if not (_has(session, "listing") and _has(session, "listing_snapshot")):
        return pd.DataFrame()
    end = datetime.combine(as_of + timedelta(days=1), time.min)
    rows = session.execute(text(
        "SELECT l.goods_type, s.price, COALESCE(l.seller_type, 'retail') FROM listing l "
        "JOIN listing_snapshot s ON s.listing_id = l.id "
        "WHERE l.goods_type IS NOT NULL AND s.price > 0 AND s.observed_at >= :a AND s.observed_at < :b"),
        {"a": end - timedelta(days=days), "b": end}).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["goods_type", "price", "seller"])
    df["seller"] = np.where(df.seller == "resale", "resale", "retail")

    def stats(d):
        g = d.groupby("goods_type").price
        out = pd.DataFrame({"n": g.size(), "ref": g.median(), "lo": g.quantile(0.1), "hi": g.quantile(0.9)})
        return out[out.n >= min_n]

    retail = stats(df[df.seller == "retail"]).assign(source="retail", resale_ratio=np.nan)
    resale = stats(df[df.seller == "resale"])
    both = retail.index.intersection(resale.index)
    ratio = float(np.median(retail.loc[both, "ref"] / resale.loc[both, "ref"])) if len(both) >= MIN_OVERLAP_TYPES else None
    fill = resale.drop(index=retail.index, errors="ignore").copy()
    if not fill.empty:
        if ratio is not None:
            fill[["ref", "lo", "hi"]] *= ratio
            fill["resale_ratio"] = ratio
        else:
            fill["lo"] = fill["lo"] * 0.5
            fill["ref"] = np.sqrt(fill["lo"] * fill["hi"])
            fill["resale_ratio"] = np.nan
        fill["source"] = "resale"
    return pd.concat([retail, fill])
