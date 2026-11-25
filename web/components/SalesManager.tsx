"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { decode, mapHeaders, parseCsv, TEMPLATE_HEADER, toCsv } from "@/lib/csv";

type GroupOption = { key: string; name: string; aliases: string[] };
type Sale = {
  id: number; group_key: string; goods_type: string; product_name: string; price: number; unit_cost: number | null;
  quantity_made: number; quantity_sold: number; sold_out: boolean; sale_start: string; sale_end: string | null;
  sold_out_at: string | null; channel: string;
};
type Problem = { line: number; errors: string[] };

const PW_KEY = "salesopt-pw";
const today = () => new Date(Date.now() + 9 * 3600e3).toISOString().slice(0, 10);   // 한국 날짜
const empty = () => ({
  id: 0, group_key: "", goods_type: "", product_name: "", price: "", unit_cost: "", quantity_made: "", quantity_sold: "",
  sold_out: "auto", sale_start: today(), sale_end: "", sold_out_at: "", channel: "",
});
type FormState = ReturnType<typeof empty>;
const won = (v: number) => `${Math.round(v).toLocaleString("ko-KR")}원`;
const num = (v: number) => Math.round(v).toLocaleString("ko-KR");

export default function SalesManager({ groups, goodsTypes, writable }: { groups: GroupOption[]; goodsTypes: string[]; writable: boolean }) {
  const [pw, setPw] = useState("");
  const [authed, setAuthed] = useState(false);
  const [rows, setRows] = useState<Sale[] | null>(null);
  const [form, setForm] = useState<FormState>(() => ({ ...empty(), group_key: groups[0]?.key ?? "" }));
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string; problems?: Problem[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [upload, setUpload] = useState<{ name: string; rows: Record<string, string>[]; unknown: string[]; missing: string[] } | null>(null);

  const nameOf = useMemo(() => Object.fromEntries(groups.map((g) => [g.key, g.name])), [groups]);
  const keyOf = useCallback((v: string) => {
    const q = v.replace(/\s+/g, "").toLowerCase();
    const g = groups.find((x) => [x.key, x.name, ...x.aliases].some((a) => a.replace(/\s+/g, "").toLowerCase() === q));
    return g ? g.key : v.trim();
  }, [groups]);

  const api = useCallback(async (method: string, body?: unknown, query = "") => {
    const r = await fetch(`/api/sales${query}`, {
      method, headers: { "x-access-password": pw, ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await r.json();
    if (r.status === 401) setAuthed(false);
    if (!r.ok) throw Object.assign(new Error(data.error ?? `오류 ${r.status}`), { problems: data.problems as Problem[] | undefined });
    return data;
  }, [pw]);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      const d = await api("GET");
      setRows(d.rows);
      setAuthed(true);
      try { localStorage.setItem(PW_KEY, pw); } catch { /* 무시 */ }
    } catch (e) {
      setMsg({ kind: "err", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }, [api, pw]);

  useEffect(() => {
    try { const v = localStorage.getItem(PW_KEY); if (v) setPw(v); } catch { /* 무시 */ }
  }, []);

  const set = (k: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.value });

  async function save(list: Record<string, unknown>[], done: string) {
    setBusy(true);
    setMsg(null);
    try {
      const d = await api("POST", { rows: list });
      setMsg({ kind: "ok", text: `${done}: 추가 ${d.added}건, 갱신 ${d.updated}건` });
      await load();
      return true;
    } catch (e) {
      setMsg({ kind: "err", text: (e as Error).message, problems: (e as { problems?: Problem[] }).problems });
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function submitOne(e: React.FormEvent) {
    e.preventDefault();
    const { sold_out, id, ...rest } = form;
    const ok = await save([{ ...rest, ...(id ? { id } : {}), sold_out: sold_out === "auto" ? "" : sold_out === "y" }],
      id ? "수정했습니다" : "저장했습니다");
    if (ok) setForm({ ...empty(), group_key: form.group_key, goods_type: form.goods_type, sale_start: form.sale_start });
  }

  function edit(r: Sale) {
    setForm({
      id: r.id, group_key: r.group_key, goods_type: r.goods_type, product_name: r.product_name, price: String(r.price),
      unit_cost: r.unit_cost === null ? "" : String(r.unit_cost), quantity_made: String(r.quantity_made),
      quantity_sold: String(r.quantity_sold),
      sold_out: r.sold_out === (r.quantity_sold >= r.quantity_made) ? "auto" : r.sold_out ? "y" : "n", sale_start: r.sale_start,
      sale_end: r.sale_end ?? "", sold_out_at: r.sold_out_at ?? "", channel: r.channel,
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function remove(r: Sale) {
    if (!confirm(`${nameOf[r.group_key] ?? r.group_key} · ${r.goods_type} · ${r.product_name || "(상품명 없음)"} (${r.sale_start}) 기록을 지울까요?`)) return;
    setBusy(true);
    try {
      await api("DELETE", undefined, `?id=${r.id}`);
      setMsg({ kind: "ok", text: "지웠습니다" });
      await load();
    } catch (e) {
      setMsg({ kind: "err", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    const table = parseCsv(decode(await f.arrayBuffer()));
    if (table.length < 2) { setMsg({ kind: "err", text: "CSV 에 기록이 없습니다 (첫 줄은 열 이름)" }); return; }
    const { index, unknown } = mapHeaders(table[0]);
    const need = ["group", "goods_type", "price", "quantity_made", "quantity_sold", "sale_start"];
    const missing = need.filter((k) => !(k in index));
    const parsed = table.slice(1).map((cells) => {
      const o: Record<string, string> = {};
      for (const [k, i] of Object.entries(index)) o[k] = (cells[i] ?? "").trim();
      o.group_key = keyOf(o.group ?? "");
      return o;
    });
    setUpload({ name: f.name, rows: parsed, unknown, missing });
    setMsg(null);
  }

  async function saveUpload() {
    if (!upload) return;
    const ok = await save(upload.rows.map(({ group, ...r }) => r), `${upload.name} 저장`);
    if (ok) setUpload(null);
  }

  function downloadTemplate() {
    const g = groups[0]?.name ?? "아일릿";
    const csv = toCsv([TEMPLATE_HEADER,
      [g, "키링", "아크릴 키링 A", 12000, 4000, 500, 500, "Y", "2026-10-01", "2026-10-07", "2026-10-01 20:15", "자사몰"],
      [g, "인형", "인형 B", 32000, 11000, 300, 212, "N", "2026-10-01", "2026-10-07", "", "자사몰"]]);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    a.download = "판매기록_양식.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function downloadAll() {
    if (!rows) return;
    const csv = toCsv([TEMPLATE_HEADER, ...rows.map((r) => [nameOf[r.group_key] ?? r.group_key, r.goods_type, r.product_name, r.price,
      r.unit_cost, r.quantity_made, r.quantity_sold, r.sold_out ? "Y" : "N", r.sale_start, r.sale_end, r.sold_out_at?.replace("T", " ") ?? "", r.channel])]);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    a.download = `판매기록_${today()}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  if (!writable) {
    return (
      <section className="card">
        <p>Vercel 환경변수 <code>ACCESS_PASSWORD</code> 를 설정하세요.</p>
      </section>
    );
  }

  if (!authed) {
    return (
      <form className="card form" onSubmit={(e) => { e.preventDefault(); load(); }}>
        <label className="pw"><span>비밀번호</span><input type="password" value={pw} onChange={(e) => setPw(e.target.value)} required autoFocus /></label>
        <button type="submit" disabled={busy}>{busy ? "확인 중…" : "열기"}</button>
        {msg && <p className={msg.kind === "err" ? "error" : "okmsg"} role="alert">{msg.text}</p>}
      </form>
    );
  }

  const n = rows?.length ?? 0;
  const soldOut = rows?.filter((r) => r.sold_out).length ?? 0;
  const prices = new Set(rows?.map((r) => `${r.goods_type}:${r.price}`));

  return (
    <>
      <section className="tiles small">
        <div className="tile"><span>실제 판매 기록</span><b>{num(n)}건</b></div>
        <div className="tile"><span>완판</span><b>{num(soldOut)}건</b></div>
        <div className="tile"><span>남은 재고 있음</span><b>{num(n - soldOut)}건</b>
</div>
        <div className="tile"><span>굿즈·가격 조합</span><b>{num(prices.size)}가지</b></div>
      </section>

      <form className="card form" onSubmit={submitOne}>
        <h2 className="formtitle">{form.id ? "기록 고치기" : "기록 입력"}</h2>
        <div className="fields">
          <label><span>아이돌 그룹</span>
            <select value={form.group_key} onChange={set("group_key")} required>
              {groups.map((g) => <option key={g.key} value={g.key}>{g.name}</option>)}
            </select>
          </label>
          <label><span>굿즈 종류</span>
            <input list="types" value={form.goods_type} onChange={set("goods_type")} placeholder="예) 키링" required maxLength={50} />
            <datalist id="types">{goodsTypes.map((t) => <option key={t} value={t} />)}</datalist>
          </label>
          <label><span>상품명</span><input value={form.product_name} onChange={set("product_name")} placeholder="예) 아크릴 키링 A" maxLength={300} /></label>
          <label><span>판매가 (원)</span><input inputMode="numeric" value={form.price} onChange={set("price")} required /></label>
          <label><span>개당 원가 (원)</span><input inputMode="numeric" value={form.unit_cost} onChange={set("unit_cost")} /></label>
          <label><span>만든 수량</span><input inputMode="numeric" value={form.quantity_made} onChange={set("quantity_made")} required /></label>
          <label><span>팔린 수량</span><input inputMode="numeric" value={form.quantity_sold} onChange={set("quantity_sold")} required /></label>
          <label><span>완판 여부</span>
            <select value={form.sold_out} onChange={set("sold_out")}>
              <option value="auto">자동</option>
              <option value="y">완판</option>
              <option value="n">남음</option>
            </select>
          </label>
          <label><span>판매 시작일</span><input type="date" value={form.sale_start} onChange={set("sale_start")} required /></label>
          <label><span>판매 종료일</span><input type="date" value={form.sale_end} onChange={set("sale_end")} /></label>
          <label><span>완판 시각</span><input type="datetime-local" value={form.sold_out_at} onChange={set("sold_out_at")} /></label>
          <label><span>판매처</span><input value={form.channel} onChange={set("channel")} placeholder="예) 자사몰" maxLength={80} /></label>
        </div>
        <div className="actions">
          <button type="submit" disabled={busy}>{form.id ? "고친 내용 저장" : "저장"}</button>
          {form.id ? <button type="button" className="ghost" onClick={() => setForm({ ...empty(), group_key: form.group_key })}>취소</button> : null}
        </div>
      </form>

      <section className="card upload">
        <h2 className="formtitle">CSV 올리기</h2>
        <div className="actions">
          <label className="filebtn">파일 고르기<input type="file" accept=".csv,text/csv" onChange={onFile} hidden /></label>
          <button type="button" className="ghost" onClick={downloadTemplate}>양식 내려받기</button>
        </div>
        {upload && (
          <div className="preview">
            <p><b>{upload.name}</b>: {num(upload.rows.length)}줄
              {upload.unknown.length > 0 && <span className="muted"> · 무시하는 열: {upload.unknown.join(", ")}</span>}</p>
            {upload.missing.length > 0 ? (
              <p className="error">필수 열이 없습니다: {upload.missing.join(", ")}</p>
            ) : (
              <>
                <div className="scroll">
                  <table>
                    <thead><tr><th>줄</th><th>그룹</th><th>종류</th><th>상품명</th><th>판매가</th><th>만든</th><th>팔린</th><th>완판</th><th>시작일</th></tr></thead>
                    <tbody>
                      {upload.rows.slice(0, 8).map((r, i) => (
                        <tr key={i}><td>{i + 1}</td><td>{nameOf[r.group_key] ?? <span className="error">{r.group || "?"}</span>}</td><td>{r.goods_type}</td>
                          <td>{r.product_name}</td><td>{r.price}</td><td>{r.quantity_made}</td><td>{r.quantity_sold}</td><td>{r.sold_out || "자동"}</td><td>{r.sale_start}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {upload.rows.length > 8 && <p className="muted">… 외 {num(upload.rows.length - 8)}줄</p>}
                <div className="actions">
                  <button type="button" onClick={saveUpload} disabled={busy}>{num(upload.rows.length)}줄 저장</button>
                  <button type="button" className="ghost" onClick={() => setUpload(null)}>취소</button>
                </div>
              </>
            )}
          </div>
        )}
      </section>

      {msg && (
        <div className={msg.kind === "err" ? "error box" : "okmsg box"} role="alert">
          {msg.text}
          {msg.problems && (
            <ul>{msg.problems.slice(0, 20).map((p) => <li key={p.line}>{p.line}번째 줄: {p.errors.join(", ")}</li>)}</ul>
          )}
        </div>
      )}

      <section className="card">
        <div className="listhead">
          <h2 className="formtitle">저장된 기록</h2>
          {n > 0 && <button type="button" className="ghost" onClick={downloadAll}>CSV 로 내려받기</button>}
        </div>
        {n === 0 ? <p className="muted">아직 기록이 없습니다.</p> : (
          <div className="scroll">
            <table>
              <thead><tr><th>시작일</th><th>그룹</th><th>종류</th><th>상품명</th><th>판매가</th><th>만든</th><th>팔린</th><th>완판</th><th>판매처</th><th></th></tr></thead>
              <tbody>
                {rows!.map((r) => (
                  <tr key={r.id} className={form.id === r.id ? "picked" : ""}>
                    <td>{r.sale_start}</td><td>{nameOf[r.group_key] ?? r.group_key}</td><td>{r.goods_type}</td><td>{r.product_name}</td>
                    <td>{won(r.price)}</td><td>{num(r.quantity_made)}</td><td>{num(r.quantity_sold)}</td>
                    <td>{r.sold_out ? "완판" : "남음"}</td><td>{r.channel}</td>
                    <td className="rowbtns">
                      <button type="button" className="mini" onClick={() => edit(r)}>고치기</button>
                      <button type="button" className="mini danger" onClick={() => remove(r)}>지우기</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
