import SalesManager from "@/components/SalesManager";
import { passwordSet } from "@/lib/auth";
import { getAllGoodsTypes, getOptions } from "@/lib/data";

export const dynamic = "force-dynamic";
export const metadata = { title: "판매 기록 · 굿즈 완판 계산기" };

export default async function SalesPage() {
  const [opts, types] = await Promise.all([getOptions(), getAllGoodsTypes()]);
  return (
    <main className="wrap">
      <header className="head">
        <h1>판매 기록</h1>
      </header>
      {opts.problem ? (
        <section className="card"><p>{opts.problem}</p></section>
      ) : (
        <SalesManager groups={opts.groups} goodsTypes={types} writable={passwordSet()} />
      )}
    </main>
  );
}
