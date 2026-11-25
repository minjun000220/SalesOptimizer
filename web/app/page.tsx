import Recommender from "@/components/Recommender";
import { getOptions } from "@/lib/data";

export const dynamic = "force-dynamic";

export default async function Page() {
  const opts = await getOptions();
  return (
    <main className="wrap">
      <header className="head">
        <h1>굿즈 완판 계산기</h1>
      </header>
      {opts.problem ? (
        <section className="card"><p>{opts.problem}</p></section>
      ) : (
        <Recommender groups={opts.groups} goodsTypes={opts.goodsTypes} needsPassword={!!process.env.ACCESS_PASSWORD} />
      )}
    </main>
  );
}
