"use client";
import { useState } from "react";
import { pitchColor } from "@/lib/pitch";

type ClassAttr = { base: number; top_features: { feature: string; value: number }[]; by_group: Record<string, number> };
export type Attribution = { space: string; top_class: string; per_class: Record<string, ClassAttr> } | null;

export function AttributionPanel({ attribution, input }: { attribution: Attribution; input?: Record<string, unknown> }) {
  const classes = attribution ? Object.keys(attribution.per_class) : [];
  const [cls, setCls] = useState<string | null>(null);
  const [mode, setMode] = useState<"features" | "groups">("features");
  if (!attribution) return <div className="text-sm text-muted">This model family has no per-feature attribution.</div>;
  const c = cls && attribution.per_class[cls] ? cls : attribution.top_class;
  const a = attribution.per_class[c];
  const items = mode === "features"
    ? a.top_features.map((f) => ({ name: f.feature, value: f.value }))
    : Object.entries(a.by_group).map(([name, value]) => ({ name, value }));
  const max = Math.max(1e-9, ...items.map((x) => Math.abs(x.value)));
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="label">Class</span>
        {classes.map((k) => (
          <button key={k} onClick={() => setCls(k)} className={`px-2 py-0.5 rounded text-xs border ${k === c ? "border-white/40" : "border-edge text-muted"}`}
            style={{ color: k === c ? pitchColor(k) : undefined }}>{k}</button>
        ))}
        <span className="ml-auto flex gap-1">
          {(["features", "groups"] as const).map((m) => (
            <button key={m} onClick={() => setMode(m)} className={`px-2 py-0.5 rounded text-xs border ${mode === m ? "border-accent text-accent" : "border-edge text-muted"}`}>
              {m === "features" ? "Top features" : "By group"}
            </button>
          ))}
        </span>
      </div>
      <div className="space-y-1.5">
        {items.map((x) => (
          <div key={x.name} className="grid grid-cols-[minmax(0,10rem)_1fr_4rem] items-center gap-2 text-xs">
            <span className="truncate text-slate-300" title={x.name}>
              {x.name}
              {input && mode === "features" && x.name in input && (
                <span className="text-muted"> = {fmtVal(input[x.name])}</span>
              )}
            </span>
            <div className="relative h-3 bg-white/5 rounded">
              <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/20" />
              <div className="absolute top-0 bottom-0 rounded"
                style={{
                  left: x.value >= 0 ? "50%" : `${50 - (Math.abs(x.value) / max) * 50}%`,
                  width: `${(Math.abs(x.value) / max) * 50}%`,
                  background: x.value >= 0 ? "#22d3ee" : "#f87171",
                }} />
            </div>
            <span className="tnum text-right text-slate-300">{x.value >= 0 ? "+" : ""}{x.value.toFixed(3)}</span>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-muted mt-3">
        Contribution toward <b style={{ color: pitchColor(c) }}>{c}</b> in {attribution.space}. Positive (cyan) pushes toward the class; base = {a.base.toFixed(3)}.
        Attributions describe this model&apos;s computation, not causal effects.
      </p>
    </div>
  );
}

function fmtVal(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}
