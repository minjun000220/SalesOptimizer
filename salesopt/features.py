"""(그룹, 굿즈 종류, 날짜) → 모델 입력 특징 한 줄.

시뮬레이션(simulate.NUMERIC_FEATURES)과 같은 열 이름을 쓴다 → 모델·최적화 코드를 바꾸지 않고 실제 데이터로 교체.
미래 정보 누수 방지: 날짜 d 의 판매에는 d 이전(d 당일 제외)에 계산된 값만 쓴다.
값이 없으면 NaN → 인코더가 학습 평균으로 채운다 (예: 그룹 갤러리를 Goods 가 수집하지 않는 경우).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from salesopt.simulate import NUMERIC_FEATURES
from salesopt.sources.goods import group_type_buzz, hot_scores
from salesopt.storage import IdolSnapshot

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GroupInfo:
    key: str                       # IdolSentiments 그룹 키 (예: illit)
    goods_gallery: str | None      # Goods 아이돌 갤러리 id (예: illit). 없으면 그룹×종류 언급 = NaN
    extra_aliases: list[str]


def load_group_map(path: Path | None = None) -> dict[str, GroupInfo]:
    raw = yaml.safe_load((path or ROOT / "config" / "groups.yaml").read_text(encoding="utf-8"))
    return {k: GroupInfo(k, (v or {}).get("goods_gallery"), (v or {}).get("aliases", []))
            for k, v in (raw.get("groups") or {}).items()}


def resolve_group(query: str, group_map: dict[str, GroupInfo], session: Session | None = None) -> str:
    """'아일릿', 'ILLIT', 'illit' → 'illit'"""
    q = "".join(query.split()).lower()
    for k, g in group_map.items():
        if q == k.lower() or q in {"".join(a.split()).lower() for a in g.extra_aliases}:
            return k
    if session is not None:
        names = session.execute(select(IdolSnapshot.group_key, IdolSnapshot.name).distinct()).all()
        for k, name in names:
            if q in ("".join(name.split()).lower(), k.lower()):
                return k
    raise SystemExit(f"그룹 '{query}' 을(를) config/groups.yaml 에서 찾지 못했습니다")


def idol_features(session: Session, group_key: str, before: date) -> dict:
    snaps = session.scalars(select(IdolSnapshot).where(IdolSnapshot.group_key == group_key,
                                                       IdolSnapshot.date < before)
                            .order_by(IdolSnapshot.date.desc()).limit(40)).all()
    nan = dict.fromkeys(["favor", "interest", "controversy", "log_mentions_per_day", "favor_change_7d",
                         "member_hhi"], np.nan)
    if not snaps:
        return nan | {"idol_snapshot_date": None}
    cur = next((s for s in snaps if s.complete), snaps[0])
    week_ago = next((s for s in snaps if s.date <= cur.date - timedelta(days=7) and s.favor is not None), None)
    f = lambda v: np.nan if v is None else float(v)                          # noqa: E731
    return {
        "favor": f(cur.favor), "interest": f(cur.interest), "controversy": f(cur.controversy),
        "log_mentions_per_day": np.log(cur.mentions_per_day) if cur.mentions_per_day else np.nan,
        "favor_change_7d": (cur.favor - week_ago.favor) if (week_ago and cur.favor is not None) else np.nan,
        "member_hhi": f(cur.member_hhi),
        "idol_snapshot_date": cur.date,
    }


def goods_features(session: Session, goods_type: str, before: date, gallery: str | None,
                   _cache: dict | None = None) -> dict:
    key = ("hot", before)
    hot = _cache.get(key) if _cache is not None else None
    if hot is None:
        hot = hot_scores(session, before - timedelta(days=1))
        if _cache is not None:
            _cache[key] = hot
    row = hot.loc[goods_type] if (not hot.empty and goods_type in hot.index) else None
    buzz = group_type_buzz(session, gallery, before) if gallery else pd.Series(dtype=float)
    g = lambda k: float(row[k]) if row is not None and pd.notna(row[k]) else np.nan     # noqa: E731
    return {
        "hot_index": g("hot_index"), "buzz_z": g("buzz_z"), "sold_out_ratio": g("sold_out_ratio"),
        "group_type_buzz_7d": float(buzz.get(goods_type, 0.0)) if gallery else np.nan,
        "goods_scored_on": row["scored_on"] if row is not None else None,
    }


def build_context(session: Session, group: GroupInfo, goods_type: str, before: date, _cache=None) -> pd.DataFrame:
    """추천용 특징 1행. before = 이 날짜 이전 정보만 사용 (오늘 추천이면 내일 날짜)."""
    feats = idol_features(session, group.key, before) | goods_features(session, goods_type, before,
                                                                         group.goods_gallery, _cache)
    feats["goods_type"] = goods_type
    feats["group_key"] = group.key
    return pd.DataFrame([feats])


def build_training_frame(session: Session, sales: pd.DataFrame, group_map: dict[str, GroupInfo]) -> pd.DataFrame:
    """판매 기록마다 판매 시작일 이전의 특징을 붙인다."""
    cache: dict = {}
    rows = []
    for _, r in sales.iterrows():
        g = group_map.get(r.group_key) or GroupInfo(r.group_key, None, [])
        ctx = build_context(session, g, r.goods_type, pd.Timestamp(r.sale_start).date(), cache).iloc[0]
        rows.append({k: ctx[k] for k in NUMERIC_FEATURES + ["idol_snapshot_date", "goods_scored_on"]})
    feats = pd.DataFrame(rows, index=sales.index)
    return pd.concat([sales, feats], axis=1)


def feature_coverage(df: pd.DataFrame) -> pd.Series:
    """특징별로 값이 있는 비율. 너무 낮은 특징은 학습에서 빼는 것이 낫다."""
    return df[NUMERIC_FEATURES].notna().mean()


def select_candidates(session: Session, group: GroupInfo, before: date, k: int = 5) -> list[str]:
    """후보 굿즈 종류: Goods 핫지수 + 이 그룹 갤러리에서의 언급 수를 순위로 합쳐 상위 k개."""
    hot = hot_scores(session, before - timedelta(days=1))
    if hot.empty:
        return []
    score = hot.hot_index.rank(pct=True)
    if group.goods_gallery:
        buzz = group_type_buzz(session, group.goods_gallery, before).reindex(hot.index).fillna(0)
        score = score + buzz.rank(pct=True)
    return list(score.sort_values(ascending=False).index[:k])
