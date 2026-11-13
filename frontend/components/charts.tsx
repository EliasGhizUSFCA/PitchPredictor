"use client";
import {
  Bar, BarChart, CartesianGrid, ComposedChart, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Scatter,
  Tooltip, XAxis, YAxis, ReferenceLine,
} from "recharts";
import { pct, pitchColor } from "@/lib/pitch";

export const axis = { stroke: "#64748b", fontSize: 11 };
export const grid = <CartesianGrid stroke="#1e2633" strokeDasharray="3 3" />;
export const tooltipStyle = {
  contentStyle: { background: "#11161e", border: "1px solid #232b38", borderRadius: 8, fontSize: 12 },
  labelStyle: { color: "#cbd5e1" },
};

type UsageRow = { key: string; n: number; [k: string]: number | string | Record<string, number> };

export function UsageStack({ rows, classes, height = 220, xLabel }: { rows: UsageRow[]; classes: string[]; height?: number; xLabel?: string }) {
  if (!rows?.length) return <div className="text-sm text-muted py-6 text-center">No pitches match these filters.</div>;
  const data = rows.map((r) => ({ ...r, label: `${r.key}` }));
  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
          {grid}
          <XAxis dataKey="label" {...axis} />
          <YAxis {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} domain={[0, 1]} />
          <Tooltip {...tooltipStyle} formatter={(v: number, name: string) => [pct(v), name]}
            labelFormatter={(l, p) => `${xLabel ? xLabel + " " : ""}${l} · n=${p?.[0]?.payload?.n ?? "?"}`} />
          {classes.map((c, i) => <Bar key={c} dataKey={c} stackId="a" fill={pitchColor(c, i)} isAnimationActive={false} />)}
        </BarChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted tnum pl-6">
        {rows.map((r) => <span key={r.key}>{r.key}: n={r.n}</span>)}
      </div>
    </div>
  );
}

export function Heat({ value, label, sub, color = "34,211,238" }: { value: number; label: React.ReactNode; sub?: React.ReactNode; color?: string }) {
  return (
    <div className="rounded-md px-1 py-1.5 text-center border border-edge/50" style={{ background: `rgba(${color},${Math.min(0.85, value * 0.9)})` }}>
      <div className="text-xs font-semibold tnum">{label}</div>
      {sub && <div className="text-[10px] text-slate-200/80 tnum">{sub}</div>}
    </div>
  );
}

export function MatrixGrid({ rows, cols, matrix, rowTitle, colTitle, normalize = "row" }: {
  rows: string[]; cols: string[]; matrix: number[][]; rowTitle: string; colTitle: string; normalize?: "row" | "none";
}) {
  return (
    <div className="overflow-x-auto">
      <div className="text-[11px] text-muted mb-1">rows: {rowTitle} · cols: {colTitle} · {normalize === "row" ? "cell % = share of row" : "counts"}</div>
      <div className="grid gap-1" style={{ gridTemplateColumns: `3rem repeat(${cols.length}, minmax(3.5rem, 1fr)) 3.5rem` }}>
        <div />
        {cols.map((c, i) => <div key={c} className="text-center text-xs font-semibold" style={{ color: pitchColor(c, i) }}>{c}</div>)}
        <div className="text-center text-[11px] text-muted">n</div>
        {rows.map((r, i) => {
          const tot = matrix[i].reduce((a, b) => a + b, 0);
          return [
            <div key={r} className="text-xs font-semibold flex items-center" style={{ color: pitchColor(r, i) }}>{r}</div>,
            ...matrix[i].map((v, j) => (
              <Heat key={`${r}-${j}`} value={tot ? v / tot : 0} label={normalize === "row" ? (tot ? pct(v / tot, 0) : "—") : v} sub={normalize === "row" ? v : undefined}
                color={i === j && rowTitle.startsWith("actual") ? "74,222,128" : undefined} />
            )),
            <div key={`${r}-n`} className="text-xs tnum text-muted flex items-center justify-center">{tot}</div>,
          ];
        })}
      </div>
    </div>
  );
}

type Bin = { lo: number; hi: number; n: number; mean_conf?: number; accuracy?: number; mean_pred?: number; observed?: number };
export function Reliability({ bins, perClass, height = 240 }: { bins?: Bin[]; perClass?: Record<string, Bin[]>; height?: number }) {
  const series: { name: string; color: string; data: { x: number; y: number; n: number }[] }[] = [];
  if (bins) series.push({ name: "Top-label", color: "#e5e7eb", data: bins.map((b) => ({ x: b.mean_conf!, y: b.accuracy!, n: b.n })) });
  if (perClass) Object.entries(perClass).forEach(([c, bs], i) => series.push({ name: c, color: pitchColor(c, i), data: bs.map((b) => ({ x: b.mean_pred!, y: b.observed!, n: b.n })) }));
  if (!series.length || series.every((s) => !s.data.length)) return <div className="text-sm text-muted">No calibration data.</div>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart margin={{ top: 8, right: 12, left: -12, bottom: 4 }}>
        {grid}
        <XAxis type="number" dataKey="x" domain={[0, 1]} {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} name="predicted" />
        <YAxis type="number" dataKey="y" domain={[0, 1]} {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} name="observed" />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="4 4" />
        <Tooltip {...tooltipStyle} formatter={(v: number) => pct(v)} labelFormatter={() => ""}
          content={({ payload }) => {
            const p = payload?.[0]?.payload;
            return p ? <div className="card px-2 py-1 text-xs tnum">pred {pct(p.x)} · obs {pct(p.y)} · n={p.n}</div> : null;
          }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map((s) => <Line key={s.name} data={s.data} dataKey="y" name={s.name} stroke={s.color} dot={{ r: 3 }} isAnimationActive={false} />)}
      </LineChart>
    </ResponsiveContainer>
  );
}

type Breakdown = { value: string; n: number; top1_accuracy: number; ci95: number[]; log_loss: number; brier: number };
export function BreakdownChart({ rows, height = 220 }: { rows: Breakdown[]; height?: number }) {
  if (!rows?.length) return <div className="text-sm text-muted">No data.</div>;
  const data = rows.map((r) => ({ ...r, err: [r.top1_accuracy - r.ci95[0], r.ci95[1] - r.top1_accuracy] }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        {grid}
        <XAxis dataKey="value" {...axis} interval={0} angle={rows.length > 10 ? -40 : 0} textAnchor={rows.length > 10 ? "end" : "middle"} height={rows.length > 10 ? 50 : 30} />
        <YAxis {...axis} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip {...tooltipStyle} content={({ payload }) => {
          const p = payload?.[0]?.payload as Breakdown | undefined;
          return p ? <div className="card px-2 py-1 text-xs tnum">{p.value}: {pct(p.top1_accuracy)} (95% CI {pct(p.ci95[0])}–{pct(p.ci95[1])}) · n={p.n} · LL {p.log_loss.toFixed(3)}</div> : null;
        }} />
        <Bar dataKey="top1_accuracy" fill="#22d3ee" fillOpacity={0.7} isAnimationActive={false}>
          <ErrorBar dataKey="err" width={4} stroke="#e2e8f0" strokeWidth={1} />
        </Bar>
        <Scatter dataKey="top1_accuracy" fill="transparent" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function ProbLines({ points, classes, xKey = "value", height = 260, xLabel }: {
  points: { value: string | number; probabilities: Record<string, number> }[]; classes: string[]; xKey?: string; height?: number; xLabel?: string;
}) {
  const data = points.map((p) => ({ [xKey]: p.value, ...p.probabilities }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 4 }}>
        {grid}
        <XAxis dataKey={xKey} {...axis} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 11 } : undefined} />
        <YAxis {...axis} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip {...tooltipStyle} formatter={(v: number, n: string) => [pct(v), n]} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {classes.map((c, i) => <Line key={c} dataKey={c} stroke={pitchColor(c, i)} strokeWidth={2} dot={false} isAnimationActive={false} />)}
      </LineChart>
    </ResponsiveContainer>
  );
}
