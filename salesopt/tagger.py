"""상품 제목 → 아이돌 그룹. IdolSentiments 의 config/groups/*.yaml 별칭을 그대로 쓴다.

Goods 는 아티스트를 구분하지 않으므로 ("BTS 응원봉" → '응원봉'), 그룹별 수요를 보려면 제목에서 그룹을 다시 찾아야 한다.
- 그룹 이름(별칭)만 쓴다. 멤버 이름은 여러 그룹·일반 단어와 겹치기 쉬워 제외.
- 한 제목에서 그룹이 두 개 이상 잡히면 (콜라보 등) None.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

IDOL_CONFIG_SOURCE = "https://raw.githubusercontent.com/minjun000220/IdolSentiments/main/config/groups"


def _pattern(alias: str) -> str:
    esc = re.escape(alias)
    if re.search(r"[가-힣]", alias):
        return rf"(?<![가-힣]){esc}"                     # 왼쪽에 한글이 붙으면 다른 단어 ("아일릿" ⊂ "?아일릿" 방지)
    return rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])"


class GroupTagger:
    def __init__(self, aliases: dict[str, list[str]]):
        self.patterns = {}
        for key, al in aliases.items():
            al = sorted({a for a in al if a and len(a.strip()) >= 2}, key=len, reverse=True)
            if al:
                self.patterns[key] = re.compile("|".join(_pattern(a) for a in al), re.IGNORECASE)

    def tag(self, title: str) -> str | None:
        hits = [k for k, p in self.patterns.items() if p.search(title or "")]
        return hits[0] if len(hits) == 1 else None

    @classmethod
    def from_idol_configs(cls, source: str, keys: list[str] | None = None,
                          extra: dict[str, list[str]] | None = None) -> "GroupTagger":
        """source: IdolSentiments 의 config/groups 폴더 경로 또는 raw GitHub 주소."""
        aliases: dict[str, list[str]] = {}
        if source.startswith("http"):
            import httpx
            from urllib.parse import quote
            for k in keys or []:
                r = httpx.get(f"{source.rstrip('/')}/{quote(k)}.yaml", timeout=30, follow_redirects=True)
                r.raise_for_status()
                aliases[k] = (yaml.safe_load(r.text).get("group") or {}).get("aliases", [])
        else:
            for p in sorted(Path(source).glob("*.yaml")):
                if keys and p.stem not in keys:
                    continue
                raw = yaml.safe_load(p.read_text(encoding="utf-8"))
                g = raw.get("group") or {}
                aliases[p.stem] = [g.get("name", "")] + g.get("aliases", [])
        for k, v in (extra or {}).items():
            aliases.setdefault(k, []).extend(v)
        return cls(aliases)
