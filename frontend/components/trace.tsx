"use client";
import { useState } from "react";
import { Badge, JsonInspector } from "./ui";

export type Stage = { stage: string; status: string; started_at: string; duration_ms: number; inputs: unknown; outputs: unknown;
  warnings: string[]; error: string | null; model_version?: string };
export type Trace = { trace_id: string; kind: string; model_version: string | null; stages: Stage[]; total_ms: number; created_at?: string };

export function TraceTimeline({ trace }: { trace: Trace }) {
  const [open, setOpen] = useState<number | null>(null);
  const max = Math.max(1, ...trace.stages.map((s) => s.duration_ms));
  return (
    <div>
      <div className="flex flex-wrap gap-2 items-center text-xs text-muted mb-2">
        <span className="font-mono">{trace.trace_id}</span><Badge>{trace.kind}</Badge>
        <span className="tnum">total {trace.total_ms.toFixed(1)} ms</span>
        {trace.model_version && <span className="font-mono truncate">{trace.model_version}</span>}
      </div>
      <ol className="space-y-1">
        {trace.stages.map((s, i) => (
          <li key={i} className="rounded-md border border-edge bg-panel2">
            <button className="w-full grid grid-cols-[1.5rem_minmax(0,1fr)_5rem_minmax(3rem,8rem)] items-center gap-2 px-2 py-1.5 text-left text-xs" onClick={() => setOpen(open === i ? null : i)}>
              <span className="text-muted tnum">{i + 1}</span>
              <span className="truncate"><span className="text-slate-200">{s.stage.replaceAll("_", " ")}</span> <Badge tone={s.status}>{s.status}</Badge></span>
              <span className="tnum text-right">{s.duration_ms.toFixed(2)} ms</span>
              <span className="h-1.5 bg-white/5 rounded"><span className="block h-full rounded bg-accent/70" style={{ width: `${(s.duration_ms / max) * 100}%` }} /></span>
            </button>
            {open === i && (
              <div className="px-3 pb-3 space-y-2">
                <div className="text-[11px] text-muted tnum">started {s.started_at}</div>
                {s.warnings?.length > 0 && <div className="text-xs text-amber-300">⚠ {s.warnings.join("; ")}</div>}
                {s.error && <div className="text-xs text-red-300">✖ {s.error}</div>}
                <JsonInspector data={s.inputs} label="inputs" />
                <JsonInspector data={s.outputs} label="outputs" defaultOpen />
              </div>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
