import { passwordOk, passwordSet } from "@/lib/auth";
import { getContext, getModel } from "@/lib/data";
import { priceGrid, recommend, type Cost } from "@/lib/optimize";

export const dynamic = "force-dynamic";

// 계산은 ACCESS_PASSWORD 가 있을 때만 비밀번호 확인 (없으면 누구나)
const allowed = (req: Request) => !passwordSet() || passwordOk(req);

const bad = (msg: string, status = 400) => Response.json({ error: msg }, { status });

function num(v: unknown, name: string, { min = 0, max = 1e9, optional = false } = {}): number | null {
  if (v === undefined || v === null || v === "") {
    if (optional) return null;
    throw new Error(`${name}을(를) 입력하세요`);
  }
  const n = typeof v === "number" ? v : Number(String(v).replace(/[,_\s원]/g, ""));
  if (!Number.isFinite(n) || n < min || n > max) throw new Error(`${name} 값이 올바르지 않습니다`);
  return n;
}

const FEATURE_LABELS: Record<string, string> = {
  favor: "그룹 호감도",
  interest: "그룹 관심도",
  controversy: "논란도",
  log_mentions_per_day: "하루 언급 수 (log)",
  favor_change_7d: "호감도 7일 변화",
  member_hhi: "멤버 언급 쏠림도",
  hot_index: "굿즈 핫지수",
  buzz_z: "굿즈 언급 증가 (z)",
  sold_out_ratio: "굿즈 품절 비율",
  group_type_buzz_7d: "이 그룹 갤러리의 언급 수 (7일)",
};

export async function POST(req: Request) {
  if (!allowed(req)) return bad("비밀번호가 틀렸습니다", 401);
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return bad("잘못된 요청");
  }
  try {
    const group = String(body.group ?? "").trim().slice(0, 60);
    const goodsType = String(body.goodsType ?? "").trim().slice(0, 40);
    if (!group) return bad("아이돌 그룹 이름을 입력하세요");
    if (!goodsType) return bad("굿즈 종류를 고르세요");
    const cost: Cost = {
      unit_cost: num(body.unitCost, "개당 원가", { min: 1, max: 1e8 })!,
      salvage: num(body.salvage, "처분가", { optional: true, max: 1e8 }) ?? 0,
      fixed_cost: num(body.fixedCost, "고정비", { optional: true, max: 1e11 }) ?? 0,
      moq: Math.round(num(body.moq, "최소 생산 수량", { optional: true, min: 1, max: 1e7 }) ?? 1),
      max_qty: num(body.maxQty, "최대 생산 수량", { optional: true, min: 1, max: 1e8 }),
    };
    if (cost.salvage >= cost.unit_cost) return bad("처분가는 원가보다 작아야 합니다");
    const target = num(body.sellout, "목표 완판 확률", { optional: true, min: 50, max: 99.5 }) ?? 90;
    const alpha = 1 - target / 100;

    const ctx = await getContext(group, goodsType);
    if (ctx === "no-group") return bad(`'${group}' 그룹을 찾지 못했습니다`, 404);
    if (ctx === "no-type") return bad(`'${goodsType}' 은(는) 아직 계산할 수 없는 굿즈 종류입니다`, 422);

    const real = await getModel("real");
    const useReal = real && real.categories.includes(goodsType);
    const model = useReal ? real : await getModel("demo");
    if (!model) return bad("모델이 아직 없습니다", 503);
    if (!model.categories.includes(goodsType))
      return bad(`'${goodsType}' 은(는) 계산할 수 없는 굿즈 종류입니다`, 422);

    const support = model.price_support[goodsType];
    let lo = num(body.priceMin, "최저 판매가", { optional: true, min: 1, max: 1e9 });
    let hi = num(body.priceMax, "최고 판매가", { optional: true, min: 1, max: 1e9 });
    if (lo === null && hi === null) {
      [lo, hi] = support;
    } else {
      lo = lo ?? hi!;
      hi = hi ?? lo;
      if (lo > hi) [lo, hi] = [hi, lo];
    }
    if (hi <= cost.unit_cost) return bad("판매가가 원가보다 높아야 합니다");
    const prices = priceGrid(Math.max(lo, cost.unit_cost + 1), hi);
    const result = recommend(model, ctx.features, goodsType, prices, cost, alpha);

    const basis = model.numeric.map((f) => ({ key: f, label: FEATURE_LABELS[f] ?? f, value: ctx.features[f] ?? null }));
    return Response.json({
      group: { key: ctx.groupKey, name: ctx.groupName },
      goodsType,
      productName: String(body.productName ?? "").slice(0, 80),
      target,
      model: { kind: model.kind, meta: model.meta, members: model.members.length },
      asOf: ctx.asOf,
      idolSnapshotDate: ctx.features.idol_snapshot_date ?? null,
      level: result.level,
      best: result.best,
      rows: result.rows,
      basis,
    });
  } catch (e) {
    return bad((e as Error).message);
  }
}
