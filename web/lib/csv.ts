/** 브라우저용 CSV 읽기 (외부 의존성 없음). 따옴표·쉼표·줄바꿈이 든 칸, UTF-8 BOM, 엑셀 기본(CP949) 인코딩 지원. */

export function decode(buf: ArrayBuffer): string {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buf).replace(/^﻿/, "");
  } catch {
    return new TextDecoder("euc-kr").decode(buf); // 엑셀 '쉼표로 분리' 저장 = CP949
  }
}

export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') q = false;
      else cell += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); cell = "";
      if (row.some((v) => v.trim() !== "")) rows.push(row);
      row = [];
    } else cell += c;
  }
  row.push(cell);
  if (row.some((v) => v.trim() !== "")) rows.push(row);
  return rows;
}

/** 열 이름 → 표준 이름. 한글·영문 모두 받는다. */
const HEADERS: Record<string, string[]> = {
  group: ["group", "group_key", "그룹", "아이돌", "아이돌그룹"],
  goods_type: ["goods_type", "굿즈종류", "종류", "굿즈"],
  product_name: ["product_name", "상품명", "상품", "제품명"],
  price: ["price", "판매가", "가격"],
  unit_cost: ["unit_cost", "원가", "개당원가"],
  quantity_made: ["quantity_made", "만든수량", "생산수량", "제작수량"],
  quantity_sold: ["quantity_sold", "팔린수량", "판매수량", "판매량"],
  sold_out: ["sold_out", "완판", "완판여부"],
  sale_start: ["sale_start", "판매시작일", "시작일", "판매일"],
  sale_end: ["sale_end", "판매종료일", "종료일"],
  sold_out_at: ["sold_out_at", "완판시각", "완판시간"],
  channel: ["channel", "판매처", "채널"],
};
const norm = (s: string) => s.replace(/[\s()\-_/·]/g, "").toLowerCase();

export function mapHeaders(header: string[]): { index: Record<string, number>; unknown: string[] } {
  const index: Record<string, number> = {};
  const unknown: string[] = [];
  header.forEach((h, i) => {
    const key = Object.entries(HEADERS).find(([, names]) => names.some((n) => norm(n) === norm(h)))?.[0];
    if (key && !(key in index)) index[key] = i;
    else if (h.trim()) unknown.push(h);
  });
  return { index, unknown };
}

export const TEMPLATE_HEADER = ["group", "goods_type", "product_name", "price", "unit_cost", "quantity_made",
  "quantity_sold", "sold_out", "sale_start", "sale_end", "sold_out_at", "channel"];

export function toCsv(rows: (string | number | null)[][]): string {
  const esc = (v: string | number | null) => {
    const s = v === null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return "﻿" + rows.map((r) => r.map(esc).join(",")).join("\n");
}
