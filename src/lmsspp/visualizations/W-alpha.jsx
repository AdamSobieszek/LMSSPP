import React, { useMemo, useState } from "react";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function fmt(x: number) {
  if (Math.abs(x) < 1e-10) return "0";
  if (Math.abs(x - Math.round(x)) < 1e-10) return String(Math.round(x));
  return x.toFixed(2);
}

function sampleCurve(alpha: number, pMax = 4, steps = 220) {
  const pts: { p: number; w: number }[] = [];
  const eps = 1e-3;

  if (Math.abs(alpha - 1) < 1e-6) {
    return pts;
  }

  for (let i = 0; i <= steps; i++) {
    const p = eps + (pMax - eps) * (i / steps);
    let w: number;

    if (Math.abs(alpha - 2) < 1e-6) {
      w = Math.log(p);
    } else {
      const r = Math.pow(Math.abs(1 - alpha) * p, 1 / (1 - alpha));
      w = (Math.pow(r, 2 - alpha) - 1) / ((2 - alpha) * (1 - alpha));
    }

    if (Number.isFinite(w)) pts.push({ p, w });
  }
  return pts;
}

function curveBounds(points: { p: number; w: number }[], fallback = { min: -2, max: 2 }) {
  if (!points.length) return fallback;
  let min = Infinity;
  let max = -Infinity;
  for (const pt of points) {
    min = Math.min(min, pt.w);
    max = Math.max(max, pt.w);
  }
  const pad = 0.15 * Math.max(1, max - min);
  return { min: min - pad, max: max + pad };
}

function VectorFieldPanel({ alpha, title }: { alpha: number; title: string }) {
  const size = 320;
  const half = 2.2;
  const grid = 13;
  const arrows = [] as React.ReactNode[];
  const rings = [] as React.ReactNode[];
  const contours = [] as React.ReactNode[];
  const center = size / 2;
  const direction = alpha <= 1 ? 1 : -1;

  for (let k = 1; k <= 5; k++) {
    const rr = (k / 6) * (size * 0.44);
    rings.push(
      <circle
        key={`ring-${k}`}
        cx={center}
        cy={center}
        r={rr}
        fill="none"
        stroke="rgba(255,255,255,0.8)"
        strokeDasharray="8 8"
        strokeWidth="1.5"
      />
    );
  }

  for (let k = 1; k <= 8; k++) {
    const rr = (k / 9) * (size * 0.49);
    contours.push(
      <circle
        key={`contour-${k}`}
        cx={center}
        cy={center}
        r={rr}
        fill="none"
        stroke="rgba(0,0,0,0.18)"
        strokeWidth="1"
      />
    );
  }

  for (let i = 0; i < grid; i++) {
    for (let j = 0; j < grid; j++) {
      const x = -half + (2 * half * i) / (grid - 1);
      const y = -half + (2 * half * j) / (grid - 1);
      const r = Math.hypot(x, y);
      if (r < 0.22) continue;

      const p = (1 / Math.max(1e-6, Math.abs(1 - alpha))) * Math.pow(r, 1 - alpha);
      const ux = direction * (x / r);
      const uy = direction * (y / r);
      const mag = Math.min(0.42, 0.07 + 0.08 * Math.min(3, p));
      const x1 = center + (x / half) * (size * 0.44);
      const y1 = center - (y / half) * (size * 0.44);
      const x2 = center + ((x + ux * mag) / half) * (size * 0.44);
      const y2 = center - ((y + uy * mag) / half) * (size * 0.44);
      arrows.push(
        <g key={`a-${i}-${j}`}>
          <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(255,255,255,0.9)" strokeWidth="2" />
          <polygon
            points={`${x2},${y2} ${x2 - 6 * ux - 3 * uy},${y2 + 6 * uy - 3 * ux} ${x2 - 6 * ux + 3 * uy},${y2 + 6 * uy + 3 * ux}`}
            fill="rgba(255,255,255,0.9)"
          />
        </g>
      );
    }
  }

  const label = Math.abs(alpha - 2) < 1e-6
    ? `W = -log r,  ∇W = -x/|x|²`
    : Math.abs(alpha - 1) < 1e-6
    ? `W = r - 1,  ∇W = x/|x|`
    : `∇W = (1/(1-${fmt(alpha)})) |x|^{-α} x`;

  return (
    <div className="space-y-2">
      <div className="text-center text-xl font-semibold">{title}</div>
      <div className="text-center text-sm text-slate-600">α = {fmt(alpha)}</div>
      <div className="text-center text-sm italic text-slate-700">{label}</div>
      <svg viewBox={`0 0 ${size} ${size}`} className="w-full rounded-2xl border border-slate-300 bg-white shadow-sm">
        <defs>
          <radialGradient id={`bg-${alpha.toFixed(3)}`} cx="50%" cy="50%" r="65%">
            <stop offset="0%" stopColor="#5b147d" />
            <stop offset="55%" stopColor="#365d9d" />
            <stop offset="80%" stopColor="#2cb7a5" />
            <stop offset="100%" stopColor="#d1df3f" />
          </radialGradient>
        </defs>
        <rect x="0" y="0" width={size} height={size} fill={`url(#bg-${alpha.toFixed(3)})`} />
        {contours}
        {rings}
        {arrows}
        <circle cx={center} cy={center} r="6" fill="#ef4444" />
      </svg>
    </div>
  );
}

function CurvePanel({ alpha, title, xLabel }: { alpha: number; title: string; xLabel: string }) {
  const points = useMemo(() => sampleCurve(alpha), [alpha]);
  const width = 360;
  const height = 220;
  const margin = { left: 46, right: 16, top: 14, bottom: 36 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const pMax = 4;
  const { min, max } = curveBounds(points);

  const xScale = (p: number) => margin.left + (p / pMax) * innerW;
  const yScale = (w: number) => margin.top + ((max - w) / (max - min || 1)) * innerH;

  const path = points
    .map((pt, idx) => `${idx === 0 ? "M" : "L"}${xScale(pt.p)} ${yScale(pt.w)}`)
    .join(" ");

  const zeroY = yScale(0);

  return (
    <div className="space-y-2">
      <div className="text-center text-xl font-semibold">{title}</div>
      <div className="text-center text-sm text-slate-600">α = {fmt(alpha)}</div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded-2xl border border-slate-300 bg-white shadow-sm">
        <rect x="0" y="0" width={width} height={height} fill="#fafafa" />
        {[0, 1, 2, 3, 4].map((gx) => (
          <line key={`gx-${gx}`} x1={xScale(gx)} x2={xScale(gx)} y1={margin.top} y2={height - margin.bottom} stroke="#e5e7eb" />
        ))}
        {[-2, -1, 0, 1, 2].map((gy) => (
          <line key={`gy-${gy}`} x1={margin.left} x2={width - margin.right} y1={yScale(gy)} y2={yScale(gy)} stroke="#e5e7eb" />
        ))}
        <line x1={margin.left} x2={width - margin.right} y1={zeroY} y2={zeroY} stroke="#9ca3af" />
        <line x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} stroke="#111827" strokeWidth="1.5" />
        <line x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} stroke="#111827" strokeWidth="1.5" />
        {path && <path d={path} fill="none" stroke="#2563eb" strokeWidth="3" />}
        <circle cx={xScale(1)} cy={yScale(0)} r="4.5" fill="#ef4444" />
        <text x={width / 2} y={height - 8} textAnchor="middle" fontSize="14" fill="#111827">{xLabel}</text>
        <text x={18} y={height / 2} textAnchor="middle" fontSize="14" fill="#111827" transform={`rotate(-90 18 ${height / 2})`}>W</text>
      </svg>
    </div>
  );
}

function CriticalFieldPanel() {
  return (
    <div className="space-y-2">
      <div className="text-center text-xl font-semibold">Fixed critical field</div>
      <div className="text-center text-sm text-slate-600">α = 1</div>
      <div className="text-center text-sm italic text-slate-700">W = r - 1,  ∇W = x / |x|</div>
      <svg viewBox="0 0 320 320" className="w-full rounded-2xl border border-slate-300 bg-white shadow-sm">
        <defs>
          <radialGradient id="critical-bg" cx="50%" cy="50%" r="65%">
            <stop offset="0%" stopColor="#5b147d" />
            <stop offset="55%" stopColor="#365d9d" />
            <stop offset="80%" stopColor="#2cb7a5" />
            <stop offset="100%" stopColor="#d1df3f" />
          </radialGradient>
        </defs>
        <rect x="0" y="0" width="320" height="320" fill="url(#critical-bg)" />
        {[1, 2, 3, 4, 5].map((k) => (
          <circle key={k} cx="160" cy="160" r={k * 24} fill="none" stroke="rgba(255,255,255,0.82)" strokeDasharray="8 8" strokeWidth="1.5" />
        ))}
        {Array.from({ length: 13 }).flatMap((_, i) =>
          Array.from({ length: 13 }).map((__, j) => {
            const half = 2.2;
            const x = -half + (2 * half * i) / 12;
            const y = -half + (2 * half * j) / 12;
            const r = Math.hypot(x, y);
            if (r < 0.22) return null;
            const ux = x / r;
            const uy = y / r;
            const mag = 0.21;
            const cx = 160 + (x / half) * 140;
            const cy = 160 - (y / half) * 140;
            const dx = 160 + ((x + ux * mag) / half) * 140;
            const dy = 160 - ((y + uy * mag) / half) * 140;
            return (
              <g key={`${i}-${j}`}>
                <line x1={cx} y1={cy} x2={dx} y2={dy} stroke="rgba(255,255,255,0.9)" strokeWidth="2" />
                <polygon
                  points={`${dx},${dy} ${dx - 6 * ux - 3 * uy},${dy + 6 * uy - 3 * ux} ${dx - 6 * ux + 3 * uy},${dy + 6 * uy + 3 * ux}`}
                  fill="rgba(255,255,255,0.9)"
                />
              </g>
            );
          })
        )}
        <circle cx="160" cy="160" r="6" fill="#ef4444" />
      </svg>
    </div>
  );
}

function CriticalChartPanel() {
  return (
    <div className="space-y-2">
      <div className="text-center text-xl font-semibold">Fixed critical chart</div>
      <div className="text-center text-sm text-slate-600">α = 1</div>
      <svg viewBox="0 0 360 220" className="w-full rounded-2xl border border-slate-300 bg-white shadow-sm">
        <rect x="0" y="0" width="360" height="220" fill="#fafafa" />
        {[46, 120, 194, 268, 342].map((x, idx) => <line key={idx} x1={x} x2={x} y1="14" y2="184" stroke="#e5e7eb" />)}
        {[30, 75, 120, 165].map((y, idx) => <line key={idx} x1="46" x2="344" y1={y} y2={y} stroke="#e5e7eb" />)}
        <line x1="46" x2="344" y1="120" y2="120" stroke="#9ca3af" />
        <line x1="46" x2="46" y1="14" y2="184" stroke="#111827" strokeWidth="1.5" />
        <line x1="46" x2="344" y1="184" y2="184" stroke="#111827" strokeWidth="1.5" />
        <line x1="195" x2="195" y1="14" y2="184" stroke="#2563eb" strokeWidth="4" />
        <circle cx="195" cy="120" r="4.5" fill="#ef4444" />
        <text x="180" y="208" textAnchor="middle" fontSize="14" fill="#111827">Ṕ = (1 − α)P</text>
        <text x="18" y="110" textAnchor="middle" fontSize="14" fill="#111827" transform="rotate(-90 18 110)">W</text>
        <text x="220" y="68" textAnchor="start" fontSize="16" fill="#111827">Ṕ ≡ 1</text>
      </svg>
    </div>
  );
}

export default function RadialKernelWidget() {
  const [alpha, setAlpha] = useState(0);
  const dualAlpha = 2 - alpha;

  return (
    <div className="min-h-screen bg-slate-100 p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        <Card className="border-none shadow-xl rounded-3xl">
          <CardHeader>
            <CardTitle className="text-4xl font-serif tracking-tight">Radial kernel family in x-space and information geometry</CardTitle>
            <div className="space-y-3 pt-2">
              <div className="text-sm text-slate-700">Interactive parameter α in [0, 1]. The left column uses α, the middle column stays fixed at α = 1, and the right column uses 2 − α.</div>
              <div className="px-2">
                <Slider value={[alpha]} min={0} max={1} step={0.01} onValueChange={(v) => setAlpha(v[0] ?? 0)} />
              </div>
              <div className="flex items-center justify-between text-sm text-slate-700">
                <span>α = {fmt(alpha)}</span>
                <span>critical α = 1</span>
                <span>dual 2 − α = {fmt(dualAlpha)}</span>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-8">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <VectorFieldPanel alpha={alpha} title="Variable left field" />
              <CriticalFieldPanel />
              <VectorFieldPanel alpha={dualAlpha} title="Variable right field" />
            </div>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <CurvePanel alpha={alpha} title="Variable left chart" xLabel="P = |∇W|" />
              <CriticalChartPanel />
              <CurvePanel alpha={dualAlpha} title="Variable right chart" xLabel="P = |∇W|" />
            </div>
            <div className="rounded-2xl bg-slate-50 p-4 text-sm leading-6 text-slate-700">
              <div>For α ≈ 0 the left field is close to quadratic and the right field is close to logarithmic/inversive. As α increases toward 1, the left chart tightens toward the unit-speed critical regime while the right chart moves toward the same critical endpoint from the logarithmic side.</div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
