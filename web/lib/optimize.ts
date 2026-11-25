/**
 * 안전 완판 추천 계산 (브라우저·서버 공용, 외부 의존성 없음).
 * Python salesopt/publish.py 의 deterministic_recommend 와 같은 계산이다 → lib/optimize.test.ts 가
 * Python 이 만든 test-vectors.json 과 비교한다.
 *
 * 두 모델을 같은 형태로 다룬다: 멤버마다 log(1 + D) 공간의 '정규분포 여러 개의 혼합'
 *   단순 모델(lognormal): 멤버당 정규분포 1개
 *   딥러닝(cVAE)        : 멤버당 정규분포 64개 (잠재변수를 대표점 64개로 고정해 디코더 신경망 통과)
 * 안전 수량 = 멤버별 a 분위수의 하위 conservative 백분위,  a = min(보정 수준 또는 α, (p − c)/(p − s))
 */

export type LogNormalMember = { b0: number; b: number[]; a0: number; a: number[]; elasticity: number[]; lp_center: number[] };
/** 딥러닝(cVAE) 디코더: [가중치 행렬, 편향] × 3층, 입력·출력 표준화 값 */
export type CvaeMember = { layers: [number[][], number[]][]; lp_m: number; lp_s: number; y_m: number; y_s: number };
export type Member = LogNormalMember | CvaeMember;
export type ModelPayload = {
  version: number;
  kind: string;
  model_type?: "lognormal" | "cvae";
  numeric: string[];
  mean: number[];
  std: number[];
  categories: string[];
  members: Member[];
  conservative: number;
  conformal_level: number | null;
  price_support: Record<string, [number, number]>;
  meta: Record<string, unknown>;
  z_nodes?: number[][];
  z_weights?: number[];
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

type Mix = { mu: number[]; sg: number[]; w: number[] };   // 멤버 하나 = 정규분포 혼합

/**
 * 단순 모델. 판매 기록이 없는 굿즈 종류(g = -1) → '평균 굿즈 종류':
 *   μ = 공통 부분 + 종류 효과 평균 − 탄력성 평균 × (log p − log 보통 가격)
 *   σ = sqrt(σ_공통² + 종류 효과의 분산)
 */
function lognormalMember(model: ModelPayload, m: LogNormalMember, x: number[], g: number, price: number, priceRef?: number | null): Mix {
  const lp = Math.log(price);
  const d = model.numeric.length;
  const n = g >= 0 ? x.length : d;
  let dotB = 0;
  let dotA = 0;
  for (let i = 0; i < n; i++) {
    dotB += x[i] * m.b[i];
    dotA += x[i] * m.a[i];
  }
  if (g >= 0) {
    return {
      mu: [m.b0 + dotB - m.elasticity[g] * (lp - m.lp_center[g])],
      sg: [Math.exp(Math.min(Math.max(m.a0 + dotA, -4), 3))],
      w: [1],
    };
  }
  const tb = [0, ...m.b.slice(d)];
  const ta = [0, ...m.a.slice(d)];
  const center = priceRef ? Math.log(priceRef) : avg(m.lp_center);
  const s0 = Math.exp(Math.min(Math.max(m.a0 + dotA + avg(ta), -4), 3));
  return { mu: [m.b0 + dotB + avg(tb) - avg(m.elasticity) * (lp - center)], sg: [Math.sqrt(s0 * s0 + variance(tb))], w: [1] };
}

const silu = (v: number) => v / (1 + Math.exp(-v));

function dense(input: number[], W: number[][], b: number[], act: boolean): number[] {
  const out = new Array(b.length);
  for (let j = 0; j < b.length; j++) {
    let t = b[j];
    const row = W[j];
    for (let i = 0; i < input.length; i++) t += row[i] * input[i];
    out[j] = act ? silu(t) : t;
  }
  return out;
}

/** 딥러닝(cVAE): 대표점 z 마다 디코더 신경망(은닉 2층, SiLU)을 통과 → 정규분포 64개 */
function cvaeMember(model: ModelPayload, m: CvaeMember, x: number[], price: number): Mix {
  const lp = (Math.log(price) - m.lp_m) / m.lp_s;
  const [[W1, b1], [W2, b2], [W3, b3]] = m.layers;
  const mu: number[] = [];
  const sg: number[] = [];
  for (const z of model.z_nodes!) {
    const h1 = dense([...x, lp, ...z], W1, b1, true);
    const h2 = dense(h1, W2, b2, true);
    const o = dense(h2, W3, b3, false);
    mu.push(o[0] * m.y_s + m.y_m);
    sg.push(Math.exp(Math.min(Math.max(o[1], -5), 2)) * m.y_s);
  }
  return { mu, sg, w: model.z_weights! };
}

/** 이 모델이 이 굿즈 종류를 계산할 수 있나 (딥러닝은 학습한 종류만) */
export function supports(model: ModelPayload, goodsType: string): boolean {
  return (model.model_type ?? "lognormal") === "lognormal" || model.categories.includes(goodsType);
}

function memberMixtures(model: ModelPayload, x: number[], g: number, price: number, priceRef?: number | null): Mix[] {
  if ((model.model_type ?? "lognormal") === "cvae") {
    if (g < 0) throw new Error("딥러닝 모델은 학습한 굿즈 종류만 계산할 수 있습니다");
    return (model.members as CvaeMember[]).map((m) => cvaeMember(model, m, x, price));
  }
  return (model.members as LogNormalMember[]).map((m) => lognormalMember(model, m, x, g, price, priceRef));
}

const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;

function mixCdf(c: Mix, y: number): number {
  let t = 0;
  for (let k = 0; k < c.mu.length; k++) t += c.w[k] * normCdf((y - c.mu[k]) / c.sg[k]);
  return t;
}
function mixSf(c: Mix, q: number): number {
  const y = Math.log1p(q);
  let t = 0;
  for (let k = 0; k < c.mu.length; k++) t += c.w[k] * normSf((y - c.mu[k]) / c.sg[k]);
  return t;
}
function bounds(mix: Mix[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const c of mix) for (let k = 0; k < c.mu.length; k++) {
    lo = Math.min(lo, c.mu[k] - 9 * c.sg[k]);
    hi = Math.max(hi, c.mu[k] + 9 * c.sg[k]);
  }
  return [lo, hi];
}

/** 혼합분포(멤버 평균)의 수요 level 분위수 (이분법 80회) */
function mixtureQuantile(mix: Mix[], level: number): number {
  let [lo, hi] = bounds(mix);
  for (let k = 0; k < 80; k++) {
    const mid = 0.5 * (lo + hi);
    if (mean(mix.map((c) => mixCdf(c, mid))) < level) lo = mid;
    else hi = mid;
  }
  return Math.max(Math.floor(Math.expm1(hi)), 0);
}

/** 멤버 하나의 수요 level 분위수 (연속값). 정규분포 1개면 정확한 식, 여러 개면 이분법 */
function memberQuantile(c: Mix, level: number): number {
  if (c.mu.length === 1) return Math.max(Math.expm1(c.mu[0] + c.sg[0] * normPpf(level)), 0);
  let [lo, hi] = bounds([c]);
  for (let k = 0; k < 80; k++) {
    const mid = 0.5 * (lo + hi);
    if (mixCdf(c, mid) < level) lo = mid;
    else hi = mid;
  }
  return Math.max(Math.expm1(hi), 0);
}

function expectedSold(mix: Mix[], q: number): number {
  let total = 0;
  for (const c of mix) {
    for (let j = 0; j < c.mu.length; j++) {
      let s = 0;
      for (let k = 0; k < Z.length; k++) {
        const d = Math.max(Math.floor(Math.expm1(c.mu[j] + c.sg[j] * Z[k])), 0);
        s += Math.min(q, d) * ZW[k];
      }
      total += c.w[j] * s;
    }
  }
  return total / mix.length;
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
    const mix = memberMixtures(model, x, g, p, features._price_ref as number | null | undefined);
    const rho = (p - cost.unit_cost) / (p - cost.salvage);
    const lvl = Math.min(level0, rho);
    const qb = mix.map((c) => memberQuantile(c, lvl));
    let q = Math.floor(qb.length === 1 ? qb[0] : quantileLinear(qb, cons));
    if (cost.max_qty !== null && cost.max_qty !== undefined) q = Math.min(q, cost.max_qty);
    if (q < cost.moq) q = 0;
    const sold = q > 0 ? expectedSold(mix, q) : 0;
    const profit = q > 0 ? p * sold + cost.salvage * (q - sold) - cost.unit_cost * q - cost.fixed_cost : 0;
    const sf = mix.map((c) => (q > 0 ? mixSf(c, q) : 1));
    const qNv = mixtureQuantile(mix, rho);
    const soldNv = expectedSold(mix, qNv);
    rows.push({
      price: p,
      safe_qty: q,
      sellout_prob: q > 0 ? mean(sf) : null,
      expected_sold: sold,
      expected_leftover: q > 0 ? q - sold : 0,
      expected_profit: profit,
      model_agreement: mean(sf.map((v) => (v >= 1 - alpha ? 1 : 0))),
      demand_p10: mixtureQuantile(mix, 0.1),
      demand_p50: mixtureQuantile(mix, 0.5),
      demand_p90: mixtureQuantile(mix, 0.9),
      newsvendor_qty: qNv,
      newsvendor_sellout_prob: mean(mix.map((c) => mixSf(c, qNv))),
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
