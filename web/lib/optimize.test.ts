// Python(salesopt/publish.py deterministic_recommend) 결과와 같은지 확인.
// 데이터 갱신: 저장소 루트에서 python -m pytest tests/test_publish.py
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { normPpf, normSf, priceGrid, recommend, type ModelPayload } from "./optimize.ts";

const vec = JSON.parse(readFileSync(new URL("./test-vectors.json", import.meta.url), "utf-8"));
const close = (a: number, b: number, rel: number, abs: number, what: string) =>
  assert.ok(Math.abs(a - b) <= Math.max(abs, rel * Math.abs(b)), `${what}: ${a} vs ${b}`);

test("정규분포 함수", () => {
  close(normPpf(0.1), -1.2815515655446004, 0, 1e-6, "ppf(0.1)");
  close(normPpf(0.975), 1.959963984540054, 0, 1e-6, "ppf(0.975)");
  close(normSf(1.5), 0.0668072012688581, 0, 1e-7, "sf(1.5)");
  close(normSf(-2), 0.9772498680518208, 0, 1e-7, "sf(-2)");
});

test("Python 과 같은 추천", () => {
  const model = vec.model as ModelPayload;
  for (const c of vec.cases) {
    const r = recommend(model, c.features, c.goods_type, c.prices, c.cost, c.alpha);
    assert.equal(r.rows.length, c.expected.rows.length, c.goods_type);
    r.rows.forEach((row, i) => {
      const e = c.expected.rows[i];
      assert.equal(row.price, e.price);
      close(row.safe_qty, e.safe_qty, 0, 1, `${c.goods_type} safe_qty @${e.price}`);
      close(row.expected_profit, e.expected_profit, 1e-3, 50, `${c.goods_type} profit @${e.price}`);
      close(row.demand_p50, e.demand_p50, 0, 1, `${c.goods_type} p50 @${e.price}`);
      close(row.newsvendor_qty, e.newsvendor_qty, 0, 1, `${c.goods_type} nv @${e.price}`);
      if (e.sellout_prob !== null) close(row.sellout_prob!, e.sellout_prob, 0, 1e-4, `${c.goods_type} sellout`);
    });
    assert.equal(r.best?.price ?? null, c.expected.best?.price ?? null, `${c.goods_type} best`);
  }
});

test("Python 과 같은 추천 (딥러닝 cVAE)", () => {
  assert.ok(vec.cvae, "test-vectors.json 에 cvae 데이터가 없습니다 (pytest tests/test_publish.py 를 torch 와 함께 실행)");
  const model = vec.cvae.model as ModelPayload;
  for (const c of vec.cvae.cases) {
    const r = recommend(model, c.features, c.goods_type, c.prices, c.cost, c.alpha);
    r.rows.forEach((row, i) => {
      const e = c.expected.rows[i];
      assert.equal(row.price, e.price);
      close(row.safe_qty, e.safe_qty, 0, 1, `cvae ${c.goods_type} safe_qty @${e.price}`);
      close(row.expected_profit, e.expected_profit, 1e-3, 50, `cvae ${c.goods_type} profit @${e.price}`);
      close(row.demand_p50, e.demand_p50, 0, 1, `cvae ${c.goods_type} p50`);
      close(row.demand_p90, e.demand_p90, 0, 1, `cvae ${c.goods_type} p90`);
      if (e.sellout_prob !== null) close(row.sellout_prob!, e.sellout_prob, 0, 1e-4, `cvae ${c.goods_type} sellout`);
    });
  }
  assert.throws(() => recommend(model, {}, "우산", [10000], vec.cvae.cases[0].cost));
});

test("가격 격자", () => {
  const g = priceGrid(9000, 16000);
  assert.equal(g[0], 9000);
  assert.equal(g[g.length - 1], 16000);
  assert.ok(g.length >= 20 && g.length <= 45);
  assert.deepEqual(priceGrid(12000, 12000), [12000]);
});
