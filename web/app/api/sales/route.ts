import { passwordOk, passwordSet } from "@/lib/auth";
import { deleteSale, groupKeys, listSales, saveSales, validate, type SaleIn } from "@/lib/sales";

export const dynamic = "force-dynamic";

const bad = (msg: string, status = 400, extra: object = {}) => Response.json({ error: msg, ...extra }, { status });

function gate(req: Request): Response | null {
  if (!passwordSet())
    return bad("판매 기록을 쓰려면 Vercel 환경변수 ACCESS_PASSWORD 를 먼저 넣어야 합니다 (넣은 뒤 Redeploy).", 503);
  if (!passwordOk(req)) return bad("비밀번호가 틀렸습니다", 401);
  return null;
}

export async function GET(req: Request) {
  const g = gate(req);
  if (g) return g;
  return Response.json({ rows: await listSales() });
}

/** body: { rows: [...] } (한 줄 입력도 rows 1개). 한 줄이라도 틀리면 아무것도 저장하지 않는다. */
export async function POST(req: Request) {
  const g = gate(req);
  if (g) return g;
  let body: { rows?: unknown };
  try {
    body = await req.json();
  } catch {
    return bad("잘못된 요청");
  }
  if (!Array.isArray(body.rows) || body.rows.length === 0) return bad("저장할 기록이 없습니다");
  if (body.rows.length > 2000) return bad("한 번에 2,000줄까지 올릴 수 있습니다");
  const keys = await groupKeys();
  const problems: { line: number; errors: string[] }[] = [];
  const rows: SaleIn[] = [];
  body.rows.forEach((raw, i) => {
    const r = validate((raw ?? {}) as Record<string, unknown>, keys);
    if (r.errors.length) problems.push({ line: i + 1, errors: r.errors });
    else rows.push(r.row!);
  });
  if (problems.length) return bad(`${problems.length}줄에 문제가 있어 저장하지 않았습니다`, 422, { problems });
  const res = await saveSales(rows);
  return Response.json({ ok: true, ...res });
}

export async function DELETE(req: Request) {
  const g = gate(req);
  if (g) return g;
  const id = Number(new URL(req.url).searchParams.get("id"));
  if (!Number.isInteger(id) || id <= 0) return bad("잘못된 id");
  const n = await deleteSale(id);
  return n ? Response.json({ ok: true }) : bad("기록을 찾지 못했습니다", 404);
}
