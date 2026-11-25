import Recommender from "@/components/Recommender";
import { getOptions } from "@/lib/data";

export const dynamic = "force-dynamic";

export default async function Page() {
  const opts = await getOptions();
  const real = opts.models.find((m) => m.kind === "real");
  return (
    <main className="wrap">
      <header className="head">
        <h1>굿즈 완판 계산기</h1>
        <p className="sub">
          아이돌 그룹, 굿즈 종류, 개당 원가를 넣으면 <b>예상 판매량</b>과 <b>안전하게 완판할 수 있는 판매가·생산 수량</b>을 계산합니다.
        </p>
        <p className="status">
          {real ? (
            <span className="pill ok">실제 판매 기록 {String(real.meta.n_records ?? "?")}건으로 학습 · {real.trainedAt.slice(0, 10)}</span>
          ) : (
            <span className="pill warn">⚠ 실제 판매 기록이 없어 예시 모델(가짜 데이터)로 계산합니다</span>
          )}
        </p>
      </header>
      {opts.problem ? (
        <section className="card"><p>{opts.problem}</p></section>
      ) : (
        <Recommender groups={opts.groups} goodsTypes={opts.goodsTypes} needsPassword={!!process.env.ACCESS_PASSWORD} />
      )}
      <footer className="foot">
        아이돌 점수는 IdolSentiments, 굿즈 점수는 Goods 에서 매일 갱신됩니다. 수요 모델: Gong·Liu·Zhang (2024) 의 생성모델 newsvendor 를
        완판 확률 조건으로 바꾼 것.
      </footer>
    </main>
  );
}
