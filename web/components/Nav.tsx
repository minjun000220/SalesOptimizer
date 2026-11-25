"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Nav() {
  const path = usePathname();
  return (
    <nav className="nav">
      <Link href="/" className={path === "/" ? "on" : ""}>계산기</Link>
      <Link href="/sales" className={path.startsWith("/sales") ? "on" : ""}>판매 기록</Link>
    </nav>
  );
}
