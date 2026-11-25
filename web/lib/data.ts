import "server-only";
import { sql } from "./db";
import type { ModelPayload } from "./optimize";

export type GroupOption = { key: string; name: string; aliases: string[] };
export type ModelInfo = { kind: "real" | "demo"; trainedAt: string; meta: Record<string, unknown>; categories: string[] };
export type Options = { groups: GroupOption[]; goodsTypes: string[]; models: ModelInfo[]; problem: string | null };
export type Context = { groupKey: string; groupName: string; goodsType: string; asOf: string; features: Record<string, number | null> & { idol_snapshot_date?: string | null } };

const NOT_READY = "아직 데이터가 없습니다. GitHub Actions 의 '아이돌 점수 저장 + 웹 데이터 갱신' 을 한 번 실행하세요.";

function missingTable(e: unknown) {
  return (e as { code?: string })?.code === "42P01";
}

export async function getOptions(): Promise<Options> {
  try {
    const groups = await sql<{ key: string; name: string; aliases: string[] }[]>`
      select distinct on (group_key) group_key as key, group_name as name, aliases
      from so_context order by group_key`;
    const types = await sql<{ t: string }[]>`select distinct goods_type as t from so_context order by 1`;
    const models = await sql<{ kind: "real" | "demo"; trained_at: Date; meta: Record<string, unknown>; categories: string[] }[]>`
      select kind, trained_at, payload->'meta' as meta, payload->'categories' as categories from so_model`;
    return {
      groups: groups.map((g) => ({ ...g, aliases: g.aliases ?? [] })).sort((a, b) => a.name.localeCompare(b.name, "ko")),
      goodsTypes: types.map((t) => t.t),
      models: models.map((m) => ({ kind: m.kind, trainedAt: new Date(m.trained_at).toISOString(), meta: m.meta ?? {}, categories: m.categories ?? [] })),
      problem: groups.length ? null : NOT_READY,
    };
  } catch (e) {
    if (missingTable(e)) return { groups: [], goodsTypes: [], models: [], problem: NOT_READY };
    throw e;
  }
}

export async function getModel(kind: "real" | "demo"): Promise<ModelPayload | null> {
  const rows = await sql<{ payload: ModelPayload }[]>`select payload from so_model where kind = ${kind}`;
  return rows[0]?.payload ?? null;
}

const norm = (s: string) => s.replace(/\s+/g, "").toLowerCase();

/** 그룹 키·이름·별칭 어느 것으로든 찾는다 ('아일릿', 'ILLIT', 'illit'). */
export async function getContext(group: string, goodsType: string): Promise<Context | "no-group" | "no-type"> {
  const groups = await sql<{ group_key: string; group_name: string; aliases: string[] }[]>`
    select distinct on (group_key) group_key, group_name, aliases from so_context order by group_key`;
  const q = norm(group);
  const g = groups.find((x) => norm(x.group_key) === q || norm(x.group_name) === q || (x.aliases ?? []).some((a) => norm(a) === q));
  if (!g) return "no-group";
  const rows = await sql<{ group_key: string; group_name: string; goods_type: string; as_of: string; features: Context["features"] }[]>`
    select group_key, group_name, goods_type, to_char(as_of, 'YYYY-MM-DD') as as_of, features from so_context
    where group_key = ${g.group_key} and goods_type = ${goodsType}`;
  const r = rows[0];
  if (!r) return "no-type";
  return {
    groupKey: r.group_key,
    groupName: r.group_name,
    goodsType: r.goods_type,
    asOf: r.as_of,
    features: r.features,
  };
}
