import "server-only";
import { sql } from "./db";

/** so_sales_record (Python salesopt/storage.py SalesRecord 와 같은 테이블). */
export type SaleIn = {
  id?: number;
  group_key: string;
  goods_type: string;
  product_name: string;
  price: number;
  unit_cost: number | null;
  quantity_made: number;
  quantity_sold: number;
  sold_out: boolean;
  sale_start: string;        // YYYY-MM-DD
  sale_end: string | null;
  sold_out_at: string | null; // YYYY-MM-DDTHH:MM
  channel: string;
};
export type Sale = SaleIn & { id: number };

const DATE = /^\d{4}-\d{2}-\d{2}$/;
const DATETIME = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?$/;

function n(v: unknown): number | null {
  if (v === null || v === undefined || v === "") return null;
  const x = typeof v === "number" ? v : Number(String(v).replace(/[,_\s원개]/g, ""));
  return Number.isFinite(x) ? x : NaN;
}
function yes(v: unknown): boolean | null {
  if (v === null || v === undefined || v === "") return null;
  if (typeof v === "boolean") return v;
  return ["y", "yes", "true", "1", "o", "완판", "예"].includes(String(v).trim().toLowerCase());
}

/** 입력 한 줄 검사·정리. 문제가 있으면 오류 문장 목록. */
export function validate(raw: Record<string, unknown>, groupKeys: Set<string>): { row?: SaleIn; errors: string[] } {
  const e: string[] = [];
  const group_key = String(raw.group_key ?? "").trim();
  const goods_type = String(raw.goods_type ?? "").trim();
  const price = n(raw.price);
  const unit_cost = n(raw.unit_cost);
  const made = n(raw.quantity_made);
  const sold = n(raw.quantity_sold);
  const start = String(raw.sale_start ?? "").trim();
  const end = String(raw.sale_end ?? "").trim() || null;
  const soldOutAt = String(raw.sold_out_at ?? "").trim() || null;
  if (!groupKeys.has(group_key)) e.push(`그룹 '${group_key || "(비어 있음)"}' 을(를) 찾지 못했습니다`);
  if (!goods_type) e.push("굿즈 종류가 비어 있습니다");
  if (goods_type.length > 50) e.push("굿즈 종류가 너무 깁니다");
  if (price === null || Number.isNaN(price) || price <= 0) e.push("판매가가 올바르지 않습니다");
  if (unit_cost !== null && (Number.isNaN(unit_cost) || unit_cost < 0)) e.push("원가가 올바르지 않습니다");
  if (made === null || Number.isNaN(made) || made <= 0 || !Number.isInteger(made)) e.push("만든 수량이 올바르지 않습니다");
  if (sold === null || Number.isNaN(sold) || sold < 0 || !Number.isInteger(sold)) e.push("팔린 수량이 올바르지 않습니다");
  if (made !== null && sold !== null && sold > made) e.push("팔린 수량이 만든 수량보다 많습니다");
  if (!DATE.test(start) || Number.isNaN(Date.parse(start))) e.push("판매 시작일은 2026-10-01 형식입니다");
  if (end && !DATE.test(end)) e.push("판매 종료일은 2026-10-07 형식입니다");
  if (soldOutAt && !DATETIME.test(soldOutAt)) e.push("완판 시각은 2026-10-01 20:15 형식입니다");
  const flag = yes(raw.sold_out);
  if (flag === true && made !== null && sold !== null && sold < made) e.push("완판인데 팔린 수량이 만든 수량보다 적습니다");
  if (e.length) return { errors: e };
  return {
    errors: [],
    row: {
      ...(raw.id ? { id: Number(raw.id) } : {}),
      group_key, goods_type,
      product_name: String(raw.product_name ?? "").trim().slice(0, 300),
      price: price!, unit_cost, quantity_made: made!, quantity_sold: sold!,
      sold_out: flag ?? sold! >= made!,
      sale_start: start, sale_end: end, sold_out_at: soldOutAt ? soldOutAt.replace(" ", "T").slice(0, 16) : null,
      channel: String(raw.channel ?? "").trim().slice(0, 80),
    },
  };
}

export async function groupKeys(): Promise<Set<string>> {
  const rows = await sql<{ k: string }[]>`select distinct group_key as k from so_context`;
  return new Set(rows.map((r) => r.k));
}

export async function listSales(): Promise<Sale[]> {
  return sql<Sale[]>`
    select id, group_key, goods_type, product_name, price, unit_cost, quantity_made, quantity_sold, sold_out,
           to_char(sale_start, 'YYYY-MM-DD') as sale_start, to_char(sale_end, 'YYYY-MM-DD') as sale_end,
           to_char(sold_out_at, 'YYYY-MM-DD"T"HH24:MI') as sold_out_at, channel
    from so_sales_record where source = 'real'
    order by sale_start desc, id desc limit 2000`;
}

/** 같은 (그룹, 종류, 상품명, 시작일, 판매처) 기록이 있으면 덮어쓴다. id 가 있으면 그 기록을 고친다. */
export async function saveSales(rows: SaleIn[]): Promise<{ added: number; updated: number }> {
  let added = 0;
  let updated = 0;
  await sql.begin(async (tx) => {
    for (const r of rows) {
      const v = [r.group_key, r.goods_type, r.product_name, r.price, r.unit_cost, r.quantity_made, r.quantity_sold,
        r.sold_out, r.sale_start, r.sale_end, r.sold_out_at, r.channel] as const;
      if (r.id) {
        const res = await tx`
          update so_sales_record set group_key=${v[0]}, goods_type=${v[1]}, product_name=${v[2]}, price=${v[3]},
            unit_cost=${v[4]}, quantity_made=${v[5]}, quantity_sold=${v[6]}, sold_out=${v[7]},
            sale_start=${v[8]}::date, sale_end=${v[9]}::date, sold_out_at=${v[10]}::timestamp, channel=${v[11]}
          where id = ${r.id} and source = 'real'`;
        updated += res.count;
        continue;
      }
      const [res] = await tx<{ inserted: boolean }[]>`
        insert into so_sales_record (group_key, goods_type, product_name, price, unit_cost, quantity_made, quantity_sold,
          sold_out, sale_start, sale_end, sold_out_at, channel, source)
        values (${v[0]}, ${v[1]}, ${v[2]}, ${v[3]}, ${v[4]}, ${v[5]}, ${v[6]}, ${v[7]}, ${v[8]}::date, ${v[9]}::date,
          ${v[10]}::timestamp, ${v[11]}, 'real')
        on conflict (group_key, goods_type, product_name, sale_start, channel) do update set
          price = excluded.price, unit_cost = excluded.unit_cost, quantity_made = excluded.quantity_made,
          quantity_sold = excluded.quantity_sold, sold_out = excluded.sold_out, sale_end = excluded.sale_end,
          sold_out_at = excluded.sold_out_at, source = 'real'
        returning (xmax = 0) as inserted`;
      if (res.inserted) added++;
      else updated++;
    }
  });
  return { added, updated };
}

export async function deleteSale(id: number): Promise<number> {
  const res = await sql`delete from so_sales_record where id = ${id} and source = 'real'`;
  return res.count;
}
