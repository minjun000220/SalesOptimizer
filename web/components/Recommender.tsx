"use client";

import { useEffect, useState } from "react";
import LineChart from "./LineChart";
import type { Row } from "@/lib/optimize";

type GroupOption = { key: string; name: string; aliases: string[] };
type Basis = { key: string; label: string; value: number | null };
type Answer = {
  group: { key: string; name: string };
  goodsType: string;
  productName: string;
  target: number;
  isDemo: boolean;
  model: { kind: string; meta: Record<string, unknown>; members: number };
  asOf: string;
  idolSnapshotDate: string | null;
  level: number;
  best: Row | null;
  rows: Row[];
  basis: Basis[];
  missing: string[];
  notes: string[];
};

const won = (v: number) => `${Math.round(v).toLocaleString("ko-KR")}원`;
const qty = (v: number) => `${Math.round(v).toLocaleString("ko-KR")}개`;
const pct = (v: number | null) => (v === null ? "-" : `${Math.round(v * 100)}%`);
const compactWon = (v: number) => {
  const a = Math.abs(v);
  if (a >= 1e8) return `${(v / 1e8).toFixed(1).replace(/\.0$/, "")}억`;
  if (a >= 1e4) return `${(v / 1e4).toFixed(a >= 1e5 ? 0 : 1).replace(/\.0$/, "")}만`;
  return Math.round(v).toLocaleString("ko-KR");
};
const compactQty = (v: number) => (Math.abs(v) >= 1e4 ? `${(v / 1e4).toFixed(1).replace(/\.0$/, "")}만` : Math.round(v).toLocaleString("ko-KR"));
const fmtBasis = (b: Basis) => {
  if (b.value === null) return "없음";
  if (b.key === "controversy" || b.key === "sold_out_ratio") return `${Math.round(b.value * 100)}%`;
  if (b.key === "member_hhi" || b.key === "buzz_z" || b.key === "log_mentions_per_day") return b.value.toFixed(2);
  if (b.key === "favor_change_7d") return `${b.value > 0 ? "+" : ""}${b.value.toFixed(1)}`;
  return b.value.toLocaleString("ko-KR", { maximumFractionDigits: 1 });
};

const PW_KEY = "salesopt-pw";

export default function Recommender({ groups, goodsTypes, needsPassword }: { groups: GroupOption[]; goodsTypes: string[]; needsPassword: boolean }) {
  const [form, setForm] = useState({
    group: groups[0]?.name ?? "", goodsType: goodsTypes.includes("키링") ? "키링" : goodsTypes[0] ?? "", productName: "",
    unitCost: "", sellout: "90", priceMin: "", priceMax: "", fixedCost: "", moq: "", maxQty: "", salvage: "",
  });
  const [pw, setPw] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [ans, setAns] = useState<Answer | null>(null);

  useEffect(() => {
    try { setPw(localStorage.getItem(PW_KEY) ?? ""); } catch { /* 저장소를 못 쓰는 브라우저 */ }
  }, []);

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.value });

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const r = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(needsPassword ? { "x-access-password": pw } : {}) },
        body: JSON.stringify(form),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error ?? `오류 ${r.status}`);
      if (needsPassword) { try { localStorage.setItem(PW_KEY, pw); } catch { /* 무시 */ } }
      setAns(data);
    } catch (e2) {
      setErr((e2 as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const best = ans?.best ?? null;
  const rows = ans?.rows ?? [];

  return (
    <>
      <form className="card form" onSubmit={submit}>
        <div className="fields">
          <label>
            <span>아이돌 그룹</span>
            <input list="groups" value={form.group} onChange={set("group")} placeholder="예) 아일릿" required />
            <datalist id="groups">
              {groups.flatMap((g) => [g.name, ...g.aliases.filter((a) => a !== g.name && a !== g.key)])
                .filter((v, i, a) => a.indexOf(v) === i).map((n) => <option key={n} value={n} />)}
            </datalist>
          </label>
          <label>
            <span>굿즈 종류</span>
            <select value={form.goodsType} onChange={set("goodsType")} required>
              {goodsTypes.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <label>
            <span>상품명 <em>(선택)</em></span>
            <input value={form.productName} onChange={set("productName")} placeholder="예) 아크릴 키링 시즌2" maxLength={80} />
          </label>
          <label>
            <span>개당 원가 (원)</span>
            <input inputMode="numeric" value={form.unitCost} onChange={set("unitCost")} placeholder="예) 4000" required />
          </label>
        </div>
        <details className="more">
          <summary>세부 설정</summary>
          <div className="fields">
            <label>
              <span>목표 완판 확률</span>
              <select value={form.sellout} onChange={set("sellout")}>
                {["80", "85", "90", "95", "98"].map((v) => <option key={v} value={v}>{v}%</option>)}
              </select>
            </label>
            <label><span>최저 판매가 (원)</span><input inputMode="numeric" value={form.priceMin} onChange={set("priceMin")} placeholder="비우면 자동" /></label>
            <label><span>최고 판매가 (원)</span><input inputMode="numeric" value={form.priceMax} onChange={set("priceMax")} placeholder="비우면 자동" /></label>
            <label><span>1회 고정비 (원)</span><input inputMode="numeric" value={form.fixedCost} onChange={set("fixedCost")} placeholder="디자인·금형 등, 기본 0" /></label>
            <label><span>최소 생산 수량</span><input inputMode="numeric" value={form.moq} onChange={set("moq")} placeholder="기본 1" /></label>
            <label><span>최대 생산 수량</span><input inputMode="numeric" value={form.maxQty} onChange={set("maxQty")} placeholder="제한 없음" /></label>
            <label><span>남은 재고 처분가 (원)</span><input inputMode="numeric" value={form.salvage} onChange={set("salvage")} placeholder="기본 0" /></label>
          </div>
        </details>
        {needsPassword && (
          <label className="pw"><span>비밀번호</span><input type="password" value={pw} onChange={(e) => setPw(e.target.value)} required /></label>
        )}
        <button type="submit" disabled={busy}>{busy ? "계산 중…" : "계산하기"}</button>
        {err && <p className="error" role="alert">{err}</p>}
      </form>

      {ans && (
        <section className={`result ${busy ? "stale" : ""}`} aria-live="polite">
          {ans.isDemo && (
            <div className="banner">
              ⚠ <b>예시 결과입니다.</b> 실제 판매 기록이 아직 없어서 <b>가짜 데이터</b>로 학습한 모델로 계산했습니다.
              실제 판단에 쓰지 마세요. 판매 기록이 20건 이상 쌓이면 실제 모델로 바뀝니다.
            </div>
          )}
          <h2>
            {ans.group.name} · {ans.goodsType}{ans.productName ? ` · ${ans.productName}` : ""}
          </h2>

          {best ? (
            <>
              <div className="tiles">
                <div className="tile hero"><span>적정 판매가</span><b>{won(best.price)}</b></div>
                <div className="tile hero"><span>추천 생산 수량</span><b>{qty(best.safe_qty)}</b></div>
                <div className="tile"><span>예상 판매량 (수요)</span><b>{qty(best.demand_p50)}</b><small>대략 {qty(best.demand_p10)} ~ {qty(best.demand_p90)}</small></div>
                <div className="tile"><span>완판 확률</span><b>{pct(best.sellout_prob)}</b><small>목표 {ans.target}% 이상</small></div>
                <div className="tile"><span>기대 이익</span><b>{won(best.expected_profit)}</b><small>남는 재고 평균 {qty(best.expected_leftover)}</small></div>
              </div>
              <p className="summary">
                <b>{won(best.price)}</b>에 <b>{qty(best.safe_qty)}</b>를 만들면 약 <b>{pct(best.sellout_prob)}</b> 확률로 모두 팔립니다.
                이 가격에서 수요는 중앙값 {qty(best.demand_p50)} (10~90% 범위 {qty(best.demand_p10)}~{qty(best.demand_p90)})로 예상됩니다.
              </p>
              <p className="compare">
                참고: 기대 이익만 최대로 하면(원 논문 기준) {qty(best.newsvendor_qty)}를 만들게 되는데, 이때 완판 확률은 {pct(best.newsvendor_sellout_prob)}입니다.
              </p>
            </>
          ) : (
            <p className="summary">목표 완판 확률을 지키면서 이익이 나는 가격이 없습니다.</p>
          )}

          {rows.length > 1 && (
            <div className="charts">
              <LineChart
                title="판매가별 기대 이익 (추천 수량으로 만들었을 때)"
                x={rows.map((r) => r.price)}
                series={[{ name: "기대 이익", color: "var(--series-1)", values: rows.map((r) => (r.safe_qty > 0 ? r.expected_profit : null)) }]}
                highlightX={best?.price}
                formatX={(v) => `${compactWon(v)}원`}
                formatY={(v) => compactWon(v)}
              />
              <LineChart
                title="판매가별 수량"
                x={rows.map((r) => r.price)}
                series={[
                  { name: "추천 생산 수량", color: "var(--series-1)", values: rows.map((r) => r.safe_qty) },
                  { name: "예상 수요 (중앙값)", color: "var(--series-2)", values: rows.map((r) => r.demand_p50), dashed: true },
                ]}
                band={{ name: "예상 수요 10~90%", lo: rows.map((r) => r.demand_p10), hi: rows.map((r) => r.demand_p90) }}
                highlightX={best?.price}
                formatX={(v) => `${compactWon(v)}원`}
                formatY={(v) => compactQty(v)}
              />
            </div>
          )}

          <details className="table">
            <summary>가격별 표 보기</summary>
            <div className="scroll">
              <table>
                <thead>
                  <tr><th>판매가</th><th>추천 수량</th><th>완판 확률</th><th>기대 이익</th><th>예상 수요 (10~90%)</th><th>모델 합의</th></tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.price} className={best && r.price === best.price ? "picked" : ""}>
                      <td>{won(r.price)}</td><td>{r.safe_qty ? qty(r.safe_qty) : "만들지 않음"}</td><td>{pct(r.sellout_prob)}</td>
                      <td>{won(r.expected_profit)}</td><td>{qty(r.demand_p50)} ({qty(r.demand_p10)}~{qty(r.demand_p90)})</td><td>{pct(r.model_agreement)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>

          <div className="basis">
            <h3>계산 근거 <small>({ans.asOf} 기준 · 그룹 점수 {ans.idolSnapshotDate ?? "없음"})</small></h3>
            <ul>
              {ans.basis.map((b) => <li key={b.key}><span>{b.label}</span><b>{fmtBasis(b)}</b></li>)}
            </ul>
            {ans.missing.length > 0 && <p className="muted">값이 없는 항목({ans.missing.join(", ")})은 평균으로 계산했습니다.</p>}
            <p className="muted">
              모델: {ans.isDemo ? "예시(가짜 데이터)" : `실제 판매 기록 ${String(ans.model.meta.n_records ?? "?")}건`} · 앙상블 {ans.model.members}개 ·
              분위수 수준 {ans.level.toFixed(3)}
            </p>
          </div>
          {ans.notes.length > 0 && <ul className="notes">{ans.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
        </section>
      )}
    </>
  );
}
