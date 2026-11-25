"use client";

import { useEffect, useRef, useState } from "react";

export type Series = { name: string; color: string; values: (number | null)[]; dashed?: boolean };
type Props = {
  title: string;
  x: number[];
  series: Series[];
  band?: { lo: number[]; hi: number[]; name: string };
  highlightX?: number | null;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  height?: number;
};

function ticks(lo: number, hi: number, n = 4): number[] {
  if (hi <= lo) return [lo];
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

/** 가로축 하나(가격), 세로축 하나. 마우스를 올리면 가장 가까운 가격에 세로선 + 모든 계열 값. */
export default function LineChart({ title, x, series, band, highlightX, formatX, formatY, height = 220 }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  const m = { l: 56, r: 16, t: 12, b: 28 };
  const w = width - m.l - m.r;
  const h = height - m.t - m.b;
  const all = [...series.flatMap((s) => s.values.filter((v): v is number => v !== null)), ...(band ? [...band.lo, ...band.hi] : [])];
  let yLo = Math.min(0, ...all);
  let yHi = Math.max(...all, 1);
  const pad = (yHi - yLo) * 0.06;
  yHi += pad;
  if (yLo < 0) yLo -= pad;
  const xLo = x[0];
  const xHi = x[x.length - 1] === xLo ? xLo + 1 : x[x.length - 1];
  const sx = (v: number) => m.l + ((v - xLo) / (xHi - xLo)) * w;
  const sy = (v: number) => m.t + h - ((v - yLo) / (yHi - yLo)) * h;
  const path = (vals: (number | null)[]) =>
    vals.reduce((acc, v, i) => (v === null ? acc : acc + `${acc && vals[i - 1] !== null ? "L" : "M"}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}`), "");
  const bandPath = band
    ? band.hi.map((v, i) => `${i ? "L" : "M"}${sx(x[i])},${sy(v)}`).join("") +
      band.lo.map((_, i, a) => `L${sx(x[a.length - 1 - i])},${sy(a[a.length - 1 - i])}`).join("") + "Z"
    : "";

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget as SVGRectElement).getBoundingClientRect();
    const px = xLo + ((e.clientX - r.left) / r.width) * (xHi - xLo);
    let best = 0;
    x.forEach((v, i) => { if (Math.abs(v - px) < Math.abs(x[best] - px)) best = i; });
    setHover(best);
  };
  // 오른쪽 끝 직접 라벨: 선 위(위쪽 계열)·아래(아래쪽 계열)에 두고 서로 15px 이상 떨어지게
  const endLabels = series
    .filter((s) => s.values[x.length - 1] !== null)
    .map((s) => ({ name: s.name, y: sy(s.values[x.length - 1]!) }))
    .sort((a, b) => a.y - b.y)
    .map((l, i) => ({ ...l, y: l.y + (i === 0 ? -10 : 10) }));
  for (let i = 1; i < endLabels.length; i++)
    if (endLabels[i].y - endLabels[i - 1].y < 15) endLabels[i].y = endLabels[i - 1].y + 15;
  endLabels.forEach((l) => { l.y = Math.min(Math.max(l.y, m.t + 6), m.t + h - 4); });
  for (let i = endLabels.length - 2; i >= 0; i--)
    if (endLabels[i + 1].y - endLabels[i].y < 15) endLabels[i].y = endLabels[i + 1].y - 15;
  const hi = highlightX === null || highlightX === undefined ? -1 : x.indexOf(highlightX);
  const last = x.length - 1;

  return (
    <figure className="chart" ref={ref}>
      <figcaption>{title}</figcaption>
      {series.length > 1 && (
        <div className="legend">
          {series.map((s) => (
            <span key={s.name}><i style={{ borderColor: s.color, borderStyle: s.dashed ? "dashed" : "solid" }} />{s.name}</span>
          ))}
          {band && <span><i className="bandkey" />{band.name}</span>}
        </div>
      )}
      <div className="plot">
        <svg width={width} height={height} role="img" aria-label={title}>
          {ticks(yLo, yHi).map((t) => (
            <g key={t}>
              <line x1={m.l} x2={width - m.r} y1={sy(t)} y2={sy(t)} className={t === 0 ? "axis0" : "grid"} />
              <text x={m.l - 8} y={sy(t)} dy="0.32em" textAnchor="end" className="tick">{formatY(t)}</text>
            </g>
          ))}
          {ticks(xLo, xHi, Math.max(2, Math.floor(w / 110))).map((t) => (
            <text key={t} x={sx(t)} y={height - 8} textAnchor="middle" className="tick">{formatX(t)}</text>
          ))}
          {band && <path d={bandPath} className="band" />}
          {series.map((s) => (
            <path key={s.name} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={2}
              strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {series.length > 1 && endLabels.map((l) => (
            <text key={l.name} x={width - m.r - 4} y={l.y} dy="0.32em" textAnchor="end" className="dlabel">{l.name}</text>
          ))}
          {hi >= 0 && series.map((s) => s.values[hi] !== null && (
            <circle key={s.name} cx={sx(x[hi])} cy={sy(s.values[hi]!)} r={5} fill={s.color} className="ring" />
          ))}
          {hover !== null && (
            <g>
              <line x1={sx(x[hover])} x2={sx(x[hover])} y1={m.t} y2={m.t + h} className="cross" />
              {series.map((s) => s.values[hover] !== null && (
                <circle key={s.name} cx={sx(x[hover])} cy={sy(s.values[hover]!)} r={4} fill={s.color} className="ring" />
              ))}
            </g>
          )}
          <rect x={m.l} y={m.t} width={w} height={h} fill="transparent"
            onPointerMove={onMove} onPointerLeave={() => setHover(null)} onPointerDown={onMove} />
        </svg>
        {hover !== null && (
          <div className="tip" style={{ left: sx(x[hover]) > width * 0.55 ? undefined : sx(x[hover]) + 10, right: sx(x[hover]) > width * 0.55 ? width - sx(x[hover]) + 10 : undefined, top: m.t }}>
            <div className="tiphead">{formatX(x[hover])}</div>
            {series.map((s) => (
              <div key={s.name} className="tiprow">
                <i style={{ borderColor: s.color, borderStyle: s.dashed ? "dashed" : "solid" }} />
                <b>{s.values[hover] === null ? "-" : formatY(s.values[hover]!)}</b><span>{s.name}</span>
              </div>
            ))}
            {band && (
              <div className="tiprow"><i className="bandkey" /><b>{formatY(band.lo[hover])}~{formatY(band.hi[hover])}</b><span>{band.name}</span></div>
            )}
          </div>
        )}
      </div>
    </figure>
  );
}
