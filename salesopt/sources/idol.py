"""IdolSentiments 결과 → so_idol_snapshot (날짜별 영구 보관).

IdolSentiments 는 분석 결과를 web/public/data/ 에 JSON 으로 커밋한다.
  index.json               : 그룹 목록
  groups/<key>.json        : 그룹 점수 + 멤버 점수 + 최근 90일 추이(history)
여기서는 그 JSON 을 GitHub(raw) 또는 로컬 폴더에서 읽어 매일 한 줄씩 저장한다.
추이(history)에 있는 과거 날짜도 비어 있으면 '일부 값(complete=False)' 으로 채운다.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import select
from sqlalchemy.orm import Session

from salesopt.storage import IdolSnapshot

log = logging.getLogger(__name__)

DEFAULT_SOURCE = "https://raw.githubusercontent.com/minjun000220/IdolSentiments/main/web/public/data"


def _read(source: str, rel: str) -> dict | list:
    if source.startswith("http"):
        import httpx
        r = httpx.get(f"{source.rstrip('/')}/{quote(rel)}", timeout=30, follow_redirects=True)
        r.raise_for_status()
        return r.json()
    return json.loads((Path(source) / rel).read_text(encoding="utf-8"))


def member_hhi(members: list[dict]) -> float | None:
    """멤버 언급 쏠림도 Σ share². 1/멤버수(고르게) ~ 1(한 명에게 몰림)."""
    shares = [m.get("mention_share") for m in members if m.get("mention_share") is not None]
    return float(sum(s * s for s in shares)) if shares else None


def parse_group_page(page: dict) -> dict:
    g = page.get("group", {})
    members = page.get("members", [])
    return {
        "group_key": page["key"],
        "date": date.fromisoformat(page["updated_at"][:10]),
        "name": page.get("name", page["key"]),
        "favor": g.get("favor"),
        "interest": g.get("interest"),
        "controversy": g.get("controversy"),
        "net": g.get("net"),
        "mentions_per_day": g.get("mentions_per_day"),
        "n_mentions": (page.get("counts") or {}).get("mentions"),
        "member_hhi": member_hhi(members),
        "members": {m["name"]: {k: m.get(k) for k in ("favor", "interest", "controversy", "mention_share", "n")}
                    for m in members},
        "sentiment_model": page.get("sentiment_model"),
        "complete": True,
    }


def history_rows(page: dict) -> list[dict]:
    out = []
    for h in page.get("history", []):
        out.append({
            "group_key": page["key"], "date": date.fromisoformat(h["date"]), "name": page.get("name", page["key"]),
            "favor": h.get("favor"), "interest": h.get("interest"), "mentions_per_day": h.get("mentions_per_day"),
            "members": {k: {"favor": v} for k, v in (h.get("members") or {}).items()},
            "complete": False,
        })
    return out


def upsert(session: Session, row: dict) -> str:
    cur = session.scalar(select(IdolSnapshot).where(IdolSnapshot.group_key == row["group_key"],
                                                    IdolSnapshot.date == row["date"]))
    if cur is None:
        session.add(IdolSnapshot(**row))
        return "added"
    if row["complete"] or not cur.complete:        # 완전한 값은 덮어쓰고, 완전한 값을 일부 값으로 덮지는 않음
        for k, v in row.items():
            setattr(cur, k, v)
        return "updated"
    return "kept"


def ingest(session: Session, source: str = DEFAULT_SOURCE) -> dict:
    index = _read(source, "index.json")
    stats = {"added": 0, "updated": 0, "kept": 0, "groups": 0, "failed": []}
    for g in index.get("groups", []):
        try:
            page = _read(source, f"groups/{g['key']}.json")
        except Exception as e:                       # 한 그룹 실패가 전체를 막지 않도록
            log.warning("그룹 %s 읽기 실패: %s", g["key"], e)
            stats["failed"].append(g["key"])
            continue
        stats["groups"] += 1
        for row in history_rows(page):
            if row["date"] != date.fromisoformat(page["updated_at"][:10]):
                stats[upsert(session, row)] += 1
        stats[upsert(session, parse_group_page(page))] += 1
        session.flush()
    return stats
