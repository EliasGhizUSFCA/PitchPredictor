"use client";
import { pct, pitchColor, pitchName } from "@/lib/pitch";

export type Pred = { pitch: string; pitch_name?: string; probability: number };

export function PitchChip({ code, size = "sm" }: { code?: string | null; size?: "sm" | "lg" }) {
  if (!code) return <span className="text-muted">—</span>;
  const c = pitchColor(code);
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md font-semibold ${size === "lg" ? "px-2.5 py-1 text-base" : "px-1.5 py-0.5 text-xs"}`}
      style={{ background: `${c}22`, color: c, border: `1px solid ${c}55` }}>
      <span className="rounded-full" style={{ background: c, width: size === "lg" ? 8 : 6, height: size === "lg" ? 8 : 6 }} />
      {code}
    </span>
  );
}

export function ProbBars({ predictions, actual, compact = false }: { predictions: Pred[]; actual?: string | null; compact?: boolean }) {
  if (!predictions?.length) return null;
  const max = Math.max(...predictions.map((p) => p.probability));
  return (
    <div className="space-y-2">
      {predictions.map((p, i) => {
        const c = pitchColor(p.pitch, i);
        const isActual = actual && actual === p.pitch;
        return (
          <div key={p.pitch}>
            <div className="flex justify-between text-xs mb-1">
              <span className="flex items-center gap-2">
                <span className="font-semibold" style={{ color: c }}>{p.pitch}</span>
                {!compact && <span className="text-muted">{p.pitch_name || pitchName(p.pitch)}</span>}
                {isActual && <span className="text-[10px] px-1 rounded bg-white/10 text-white">ACTUAL</span>}
              </span>
              <span className="tnum font-medium">{pct(p.probability)}</span>
            </div>
            <div className="h-2.5 rounded-full bg-white/5 overflow-hidden">
              <div className="h-full rounded-full transition-all duration-300"
                style={{ width: `${(p.probability * 100).toFixed(2)}%`, background: c, opacity: p.probability === max ? 1 : 0.6,
                  outline: isActual ? "2px solid white" : undefined }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function PitchLegend({ classes }: { classes: string[] }) {
  return (
    <div className="flex flex-wrap gap-3 text-xs text-muted">
      {classes.map((c, i) => (
        <span key={c} className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: pitchColor(c, i) }} />{c} · {pitchName(c)}
        </span>
      ))}
    </div>
  );
}
