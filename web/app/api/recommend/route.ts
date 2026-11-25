import { passwordOk, passwordSet } from "@/lib/auth";
import { getContext, getModel } from "@/lib/data";
import { priceGrid, recommend, supports, type Cost, type ModelPayload } from "@/lib/optimize";

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

    // 수요 모델 선택: auto = 검증 기록으로 자동 선택된 모델 / lognormal = 단순 확률 모델 / cvae = 딥러닝
    // 실제 판매 기록 모델이 없으면 예시 모델. 딥러닝은 학습한 굿즈 종류만 → 아니면 단순 모델로 계산
    const choice = ["auto", "lognormal", "cvae"].includes(String(body.model)) ? String(body.model) : "auto";
    let model: ModelPayload | null;
    if (choice === "cvae") {
      model = await getModel("real_cvae");
      if (!model) return bad("딥러닝 모델은 판매 기록이 100건 이상 쌓여야 학습됩니다", 422);
    } else if (choice === "lognormal") {
      model = (await getModel("real_lognormal")) ?? (await getModel("demo"));
    } else {
      model = (await getModel("real")) ?? (await getModel("demo"));
    }
    if (!model) return bad("모델이 아직 없습니다", 503);
    let fallback = false;
    if (!supports(model, goodsType)) {
      model = (await getModel("real_lognormal")) ?? (await getModel("demo"));
      if (!model) return bad("모델이 아직 없습니다", 503);
      fallback = true;
    }

    const f = ctx.features as Record<string, number | null | undefined>;
    const goodsRange: [number, number] | null = f._price_lo && f._price_hi ? [f._price_lo, f._price_hi] : null;
    const support: [number, number] | null = model.price_support[goodsType] ?? goodsRange;
    let lo = num(body.priceMin, "최저 판매가", { optional: true, min: 1, max: 1e9 });
    let hi = num(body.priceMax, "최고 판매가", { optional: true, min: 1, max: 1e9 });
    if (lo === null && hi === null) {
      // 판매 기록도 Goods 가격 자료도 없는 종류 → 원가의 1.5~4배 (굿즈는 보통 원가의 2~3배에 판매)
      [lo, hi] = support ?? [Math.round(cost.unit_cost * 1.5), Math.round(cost.unit_cost * 4)];
    } else {
      lo = lo ?? hi!;
      hi = hi ?? lo;
      if (lo > hi) [lo, hi] = [hi, lo];
    }
    if (hi <= cost.unit_cost) return bad("판매가가 원가보다 높아야 합니다");
    const prices = priceGrid(Math.max(lo, cost.unit_cost + 1), hi);
    // 기록 없는 종류의 기준 가격: Goods 가격 자료가 없으면 검토 범위의 가운데(기하평균)
    const features = { ...ctx.features } as Record<string, number | null>;
    if (!model.categories.includes(goodsType) && !features._price_ref) features._price_ref = Math.sqrt(lo * hi);
    const result = recommend(model, features, goodsType, prices, cost, alpha);

    const basis = model.numeric.map((f) => ({ key: f, label: FEATURE_LABELS[f] ?? f, value: ctx.features[f] ?? null }));
    return Response.json({
      group: { key: ctx.groupKey, name: ctx.groupName },
      goodsType,
      productName: String(body.productName ?? "").slice(0, 80),
      target,
      model: { kind: model.kind, type: model.model_type ?? "lognormal", meta: model.meta, members: model.members.length, fallback },
      asOf: ctx.asOf,
      idolSnapshotDate: ctx.features.idol_snapshot_date ?? null,
      level: result.level,
      knownType: result.knownType,
      best: result.best,
      rows: result.rows,
      basis,
    });
  } catch (e) {
    return bad((e as Error).message);
  }
}
