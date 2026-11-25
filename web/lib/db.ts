import "server-only";
import postgres from "postgres";

// 서버에서만 실행된다. DATABASE_URL(비밀번호 포함)은 브라우저로 절대 전달되지 않는다.
// 수집기·SalesOptimizer(GitHub Actions)와 같은 Supabase "Session pooler" 주소를 쓴다.
declare global {
  // 개발 모드에서 코드가 다시 로드될 때마다 연결이 새로 생기지 않게 재사용
  // eslint-disable-next-line no-var
  var __salesOptSql: ReturnType<typeof postgres> | undefined;
}

function create() {
  const url = process.env.DATABASE_URL;
  if (!url) throw new Error("DATABASE_URL 환경변수가 없습니다 (Vercel → Settings → Environment Variables).");
  const isLocal = /@(localhost|127\.0\.0\.1)[:/]/.test(url);
  return postgres(url, {
    max: 3,                 // 서버리스 함수당 연결 수를 작게
    idle_timeout: 20,
    connect_timeout: 10,
    prepare: false,         // Supabase 풀러(pgbouncer)와 호환
    ssl: isLocal ? false : "require",
  });
}

export const sql = globalThis.__salesOptSql ?? (globalThis.__salesOptSql = create());
