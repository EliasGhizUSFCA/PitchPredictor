"use client";
import { useState } from "react";
import { ApiError } from "@/lib/api";
import { useApp } from "@/lib/context";

export function Card({ title, subtitle, right, children, className = "" }: {
  title?: React.ReactNode; subtitle?: React.ReactNode; right?: React.ReactNode; children: React.ReactNode; className?: string;
}) {
  return (
    <section className={`card p-4 min-w-0 ${className}`}>
      {(title || right) && (
        <div className="flex flex-wrap items-start justify-between gap-2 mb-3">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-slate-100">{title}</h2>}
            {subtitle && <p className="text-xs text-muted mt-0.5">{subtitle}</p>}
          </div>
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ title, subtitle, right }: { title: string; subtitle?: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3 mb-5">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="text-sm text-muted mt-1">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

export function Stat({ label, value, sub, accent }: { label: string; value: React.ReactNode; sub?: React.ReactNode; accent?: boolean }) {
  return (
    <div className="bg-panel2 border border-edge rounded-lg px-3 py-2.5 min-w-0">
      <div className="label truncate">{label}</div>
      <div className={`text-lg font-semibold tnum mt-0.5 ${accent ? "text-accent" : ""}`}>{value}</div>
      {sub && <div className="text-[11px] text-muted mt-0.5 tnum">{sub}</div>}
    </div>
  );
}

const BADGE: Record<string, string> = {
  ok: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  completed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  promoted: "bg-cyan-500/15 text-cyan-300 border-cyan-500/30",
  exact_verified: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  warning: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  strong_review: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  running: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  queued: "bg-slate-500/15 text-slate-300 border-slate-500/30",
  error: "bg-red-500/15 text-red-300 border-red-500/30",
  failed: "bg-red-500/15 text-red-300 border-red-500/30",
  unavailable: "bg-red-500/15 text-red-300 border-red-500/30",
  no_match: "bg-red-500/15 text-red-300 border-red-500/30",
};
export function Badge({ children, tone }: { children: React.ReactNode; tone?: string }) {
  const cls = BADGE[tone || String(children)] || "bg-slate-500/10 text-slate-300 border-slate-500/30";
  return <span className={`inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-medium whitespace-nowrap ${cls}`}>{children}</span>;
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted py-6 justify-center">
      <span className="h-3 w-3 rounded-full border-2 border-accent border-t-transparent animate-spin" />
      {label}
    </div>
  );
}

export function ErrorBox({ error, what }: { error: ApiError | Error | string | null; what?: string }) {
  if (!error) return null;
  const msg = typeof error === "string" ? error : error.message;
  const status = typeof error === "object" && "status" in error ? (error as ApiError).status : undefined;
  return (
    <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
      {what && <span className="font-medium">{what}: </span>}
      {status ? <span className="text-red-300/80 mr-1">[{status}]</span> : null}
      {msg}
    </div>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="text-sm text-muted py-6 text-center">{children}</div>;
}

export function Note({ children, tone = "info" }: { children: React.ReactNode; tone?: "info" | "warn" }) {
  const cls = tone === "warn" ? "border-amber-500/30 bg-amber-500/10 text-amber-100" : "border-cyan-500/20 bg-cyan-500/5 text-cyan-100/90";
  return <div className={`rounded-lg border px-3 py-2 text-xs ${cls}`}>{children}</div>;
}

export function JsonInspector({ data, label = "JSON", defaultOpen = false }: { data: unknown; label?: string; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-edge bg-black/30">
      <button onClick={() => setOpen(!open)} className="w-full text-left px-3 py-1.5 text-xs text-muted hover:text-slate-200 flex justify-between">
        <span>{open ? "▾" : "▸"} {label}</span>
        <span className="font-mono">{open ? "" : "{…}"}</span>
      </button>
      {open && (
        <pre className="px-3 pb-3 text-[11px] leading-relaxed text-slate-300 overflow-auto max-h-96 font-mono">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

export function EngOnly({ children }: { children: React.ReactNode }) {
  const { engineering } = useApp();
  return engineering ? <>{children}</> : null;
}

export function Select<T extends string | number>({ value, onChange, options, className = "" }: {
  value: T | ""; onChange: (v: string) => void; options: { value: T | ""; label: string }[]; className?: string;
}) {
  return (
    <select className={`input ${className}`} value={String(value)} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => <option key={String(o.value)} value={String(o.value)}>{o.label}</option>)}
    </select>
  );
}

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 min-w-0">
      <span className="label">{label}</span>
      {children}
    </label>
  );
}

export function Tabs({ tabs, value, onChange }: { tabs: { id: string; label: string }[]; value: string; onChange: (id: string) => void }) {
  return (
    <div className="flex flex-wrap gap-1 border-b border-edge mb-4">
      {tabs.map((t) => (
        <button key={t.id} onClick={() => onChange(t.id)}
          className={`px-3 py-2 text-sm -mb-px border-b-2 ${value === t.id ? "border-accent text-slate-100" : "border-transparent text-muted hover:text-slate-200"}`}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Table({ children }: { children: React.ReactNode }) {
  return <div className="overflow-x-auto -mx-1"><table className="w-full border-collapse mx-1">{children}</table></div>;
}
