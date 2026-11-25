import "server-only";
import { createHash, timingSafeEqual } from "node:crypto";

/** ACCESS_PASSWORD 가 없으면 계산은 누구나, 판매 기록(쓰기)은 아무도 못 한다. */
export function passwordSet(): boolean {
  return !!process.env.ACCESS_PASSWORD;
}

export function passwordOk(req: Request): boolean {
  const expected = process.env.ACCESS_PASSWORD;
  if (!expected) return false;
  const given = req.headers.get("x-access-password") ?? "";
  const a = createHash("sha256").update(given).digest();
  const b = createHash("sha256").update(expected).digest();
  return timingSafeEqual(a, b);
}
