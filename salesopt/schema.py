"""실제 판매 기록 CSV 형식 (나중에 추가).

필수: group, goods_type, price, quantity_made, quantity_sold, sale_start
선택: product_name, unit_cost, sold_out(Y/N, 비우면 판매 = 생산이면 Y), sale_end, sold_out_at, channel

- 남은 재고가 있었던 기록(완판 안 됨)도 꼭 넣는다. 완판 기록만 있으면 '어디까지가 안전한가' 를 배울 수 없다.
- 가격이 여러 수준이어야 가격 효과를 배운다.
- sale_start 는 그날 이전의 그룹·굿즈 점수와 잇는 데 쓴다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ["group", "goods_type", "price", "quantity_made", "quantity_sold", "sale_start"]
OPTIONAL = ["product_name", "unit_cost", "sold_out", "sale_end", "sold_out_at", "channel"]
TEMPLATE_ROWS = [
    {"group": "아일릿", "goods_type": "키링", "product_name": "예시) 아크릴 키링 A", "price": 12000,
     "unit_cost": 4000, "quantity_made": 500, "quantity_sold": 500, "sold_out": "Y",
     "sale_start": "2026-10-01", "sale_end": "2026-10-07", "sold_out_at": "2026-10-01 20:15", "channel": "자사몰"},
    {"group": "아일릿", "goods_type": "인형", "product_name": "예시) 인형 B", "price": 32000,
     "unit_cost": 11000, "quantity_made": 300, "quantity_sold": 212, "sold_out": "N",
     "sale_start": "2026-10-01", "sale_end": "2026-10-07", "sold_out_at": "", "channel": "자사몰"},
]


def write_template(path: Path) -> Path:
    pd.DataFrame(TEMPLATE_ROWS, columns=REQUIRED[:2] + ["product_name"] + REQUIRED[2:] + OPTIONAL[1:])\
        .to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _yes(v) -> bool | None:
    if pd.isna(v) or str(v).strip() == "":
        return None
    return str(v).strip().lower() in ("y", "yes", "true", "1", "완판", "o")


def load_sales_csv(path: Path, resolve_group=None) -> pd.DataFrame:
    """CSV → 정리된 판매 기록. resolve_group: 그룹 이름 → 그룹 키 함수 (예: '아일릿' → 'illit')."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"판매 기록 CSV 에 없는 열: {missing}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = None
    for c in ("price", "quantity_made", "quantity_sold", "unit_cost"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sale_start"] = pd.to_datetime(df.sale_start).dt.date
    errors = []
    if (df.price <= 0).any() or df.price.isna().any():
        errors.append("가격이 비었거나 0 이하인 행")
    if (df.quantity_sold > df.quantity_made).any():
        errors.append("판매 수량이 생산 수량보다 큰 행")
    if df.quantity_sold.isna().any():
        errors.append("판매 수량이 빈 행")
    if errors:
        raise ValueError("판매 기록 확인 필요: " + ", ".join(errors))
    auto = df.quantity_sold >= df.quantity_made
    df["sold_out"] = [a if (y := _yes(v)) is None else y for v, a in zip(df.sold_out, auto)]
    df["group_key"] = df.group.map(resolve_group) if resolve_group else df.group
    df["product_name"] = df.product_name.fillna("")
    df["channel"] = df.channel.fillna("")
    df["source"] = "real"
    return df
