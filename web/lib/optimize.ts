/**
 * 안전 완판 추천 계산 (브라우저·서버 공용, 외부 의존성 없음).
 * Python salesopt/publish.py 의 deterministic_recommend 와 같은 계산이다 → lib/optimize.test.ts 가
 * Python 이 만든 test-vectors.json 과 비교한다.
 *
 * 멤버 b 의 수요: log(1 + D) ~ N(μ_b, σ_b²)
 *   μ_b = b0 + x·b − e_g (log p − c_g),   σ_b = exp(clip(a0 + x·a, −4, 3))
 * 안전 수량 = 멤버별 a 분위수의 하위 conservative 백분위,  a = min(보정 수준 또는 α, (p − c)/(p − s))
 */

export type Member = { b0: number; b: number[]; a0: number; a: number[]; elasticity: number[]; lp_center: number[] };
export type ModelPayload = {
  version: number;
  kind: "real" | "demo";
  numeric: string[];
  mean: number[];
  std: number[];
  categories: string[];
  members: Member[];
  conservative: number;
  conformal_level: number | null;
  price_support: Record<string, [number, number]>;
  meta: Record<string, unknown>;
};
export type Cost = { unit_cost: number; salvage: number; fixed_cost: number; moq: number; max_qty: number | null };
export type Row = {
  price: number;
  safe_qty: number;
  sellout_prob: number | null;
  expected_sold: number;
  expected_leftover: number;
  expected_profit: number;
  model_agreement: number;
  demand_p10: number;
  demand_p50: number;
  demand_p90: number;
  newsvendor_qty: number;
  newsvendor_sellout_prob: number;
  newsvendor_profit: number;
};
export type Result = { rows: Row[]; best: Row | null; level: number; knownType: boolean };

// ------------------------------------------------------------------ 정규분포
function erfc(x: number): number {
  // Numerical Recipes erfcc (상대오차 < 1.2e-7)
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z - 1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 +
        t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}
export const normSf = (z: number) => 0.5 * erfc(z / Math.SQRT2);
export const normCdf = (z: number) => 1 - normSf(z);
const normPdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

export function normPpf(p: number): number {
  // Acklam 근사 + Halley 보정 1회
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  let x: number;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= 1 - pl) {
    const q = p - 0.5;
    const r = q * q;
    x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const e = normCdf(x) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp(0.5 * x * x);
  return x - u / (1 + 0.5 * x * u);
}

// ------------------------------------------------------------------ 보조
function quantileLinear(values: number[], q: number): number {
  const a = [...values].sort((x, y) => x - y);
  const pos = (a.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

const Z = Array.from({ length: 801 }, (_, i) => -8 + (16 * i) / 800);
const ZW = (() => {
  const w = Z.map(normPdf);
  const s = w.reduce((x, y) => x + y, 0);
  return w.map((v) => v / s);
})();

/** 특징 인코딩. 학습하지 않은 굿즈 종류면 원-핫을 0 으로 두고 g = -1 (memberParams 가 평균 종류로 처리). */
export function encode(model: ModelPayload, features: Record<string, number | null | undefined>, goodsType: string) {
  const g = model.categories.indexOf(goodsType);
  const num = model.numeric.map((f, i) => {
    const v = features[f];
    if (v === null || v === undefined || Number.isNaN(v)) return 0; // 빈 값 = 학습 평균
    return (v - model.mean[i]) / model.std[i];
  });
  const onehot = model.categories.slice(1).map((_, i) => (i === g - 1 ? 1 : 0));
  return { x: [...num, ...onehot], g };
}

const avg = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
const variance = (a: number[]) => { const m = avg(a); return avg(a.map((v) => (v - m) ** 2)); };

/**
 * 판매 기록이 없는 굿즈 종류(g = -1) → '평균 굿즈 종류':
 *   μ = 공통 부분 + 종류 효과 평균 − 탄력성 평균 × (log p − log 보통 가격)
 *   σ = sqrt(σ_공통² + 종류 효과의 분산)
 */
function memberParams(model: ModelPayload, x: number[], g: number, price: number, priceRef?: number | null) {
  const lp = Math.log(price);
  const d = model.numeric.length;
  const mu: number[] = [];
  const sg: number[] = [];
  for (const m of model.members) {
    if (g >= 0) {
      let dotB = 0;
      let dotA = 0;
      for (let i = 0; i < x.length; i++) {
        dotB += x[i] * m.b[i];
        dotA += x[i] * m.a[i];
      }
      mu.push(m.b0 + dotB - m.elasticity[g] * (lp - m.lp_center[g]));
      sg.push(Math.exp(Math.min(Math.max(m.a0 + dotA, -4), 3)));
      continue;
    }
    let dotB = 0;
    let dotA = 0;
    for (let i = 0; i < d; i++) {
      dotB += x[i] * m.b[i];
      dotA += x[i] * m.a[i];
    }
    const tb = [0, ...m.b.slice(d)];
    const ta = [0, ...m.a.slice(d)];
    const center = priceRef ? Math.log(priceRef) : avg(m.lp_center);
    mu.push(m.b0 + dotB + avg(tb) - avg(m.elasticity) * (lp - center));
    const s0 = Math.exp(Math.min(Math.max(m.a0 + dotA + avg(ta), -4), 3));
    sg.push(Math.sqrt(s0 * s0 + variance(tb)));
  }
  return { mu, sg };
}

const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;

function mixtureQuantile(mu: number[], sg: number[], level: number): number {
  let lo = Math.min(...mu.map((m, i) => m - 9 * sg[i]));
  let hi = Math.max(...mu.map((m, i) => m + 9 * sg[i]));
  for (let k = 0; k < 80; k++) {
    const mid = 0.5 * (lo + hi);
    const cdf = mean(mu.map((m, i) => normCdf((mid - m) / sg[i])));
    if (cdf < level) lo = mid;
    else hi = mid;
  }
  return Math.max(Math.floor(Math.expm1(hi)), 0);
}

function expectedSold(mu: number[], sg: number[], q: number): number {
  let total = 0;
  for (let b = 0; b < mu.length; b++) {
    let s = 0;
    for (let k = 0; k < Z.length; k++) {
      const d = Math.max(Math.floor(Math.expm1(mu[b] + sg[b] * Z[k])), 0);
      s += Math.min(q, d) * ZW[k];
    }
    total += s;
  }
  return total / mu.length;
}

// ------------------------------------------------------------------ 추천
export function recommend(
  model: ModelPayload,
  features: Record<string, number | null | undefined>,
  goodsType: string,
  prices: number[],
  cost: Cost,
  alpha = 0.1,
): Result {
  const level0 = Math.min(model.conformal_level ?? alpha, alpha);
  const cons = model.conservative ?? 0.1;
  const { x, g } = encode(model, features, goodsType);
  const rows: Row[] = [];
  for (const p of [...new Set(prices)].sort((a, b) => a - b)) {
    if (p <= cost.unit_cost) continue;
    const { mu, sg } = memberParams(model, x, g, p, features._price_ref as number | null | undefined);
    const rho = (p - cost.unit_cost) / (p - cost.salvage);
    const lvl = Math.min(level0, rho);
    const zq = normPpf(lvl);
    const qb = mu.map((m, i) => Math.max(Math.expm1(m + sg[i] * zq), 0));
    let q = Math.floor(qb.length === 1 ? qb[0] : quantileLinear(qb, cons));
    if (cost.max_qty !== null && cost.max_qty !== undefined) q = Math.min(q, cost.max_qty);
    if (q < cost.moq) q = 0;
    const sold = q > 0 ? expectedSold(mu, sg, q) : 0;
    const profit = q > 0 ? p * sold + cost.salvage * (q - sold) - cost.unit_cost * q - cost.fixed_cost : 0;
    const sf = mu.map((m, i) => (q > 0 ? normSf((Math.log1p(q) - m) / sg[i]) : 1));
    const qNv = mixtureQuantile(mu, sg, rho);
    const soldNv = expectedSold(mu, sg, qNv);
    rows.push({
      price: p,
      safe_qty: q,
      sellout_prob: q > 0 ? mean(sf) : null,
      expected_sold: sold,
      expected_leftover: q > 0 ? q - sold : 0,
      expected_profit: profit,
      model_agreement: mean(sf.map((v) => (v >= 1 - alpha ? 1 : 0))),
      demand_p10: mixtureQuantile(mu, sg, 0.1),
      demand_p50: mixtureQuantile(mu, sg, 0.5),
      demand_p90: mixtureQuantile(mu, sg, 0.9),
      newsvendor_qty: qNv,
      newsvendor_sellout_prob: mean(mu.map((m, i) => normSf((Math.log1p(qNv) - m) / sg[i]))),
      newsvendor_profit:
        qNv >= cost.moq ? p * soldNv + cost.salvage * (qNv - soldNv) - cost.unit_cost * qNv - cost.fixed_cost : 0,
    });
  }
  const ok = rows.filter((r) => r.safe_qty > 0 && r.expected_profit > 0);
  const best = ok.length ? ok.reduce((a, b) => (b.expected_profit > a.expected_profit ? b : a)) : null;
  return { rows, best, level: level0, knownType: g >= 0 };
}

/** 검토할 가격 격자: 하한~상한을 대략 40칸, 보기 좋은 간격(100/500/1000원 …)으로. */
export function priceGrid(lo: number, hi: number, cells = 40): number[] {
  if (hi <= lo) return [lo];
  const raw = (hi - lo) / cells;
  const steps = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000];
  const step = steps.find((s) => s >= raw) ?? steps[steps.length - 1];
  const out: number[] = [];
  for (let p = Math.ceil(lo / step) * step; p <= hi + 1e-9; p += step) out.push(p);
  if (out.length === 0 || out[out.length - 1] < hi) out.push(hi);
  if (out[0] > lo) out.unshift(lo);
  return out;
}
