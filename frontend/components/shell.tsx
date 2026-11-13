"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { useApp } from "@/lib/context";

const NAV = [
  { href: "/", label: "Dashboard", icon: "◉" },
  { href: "/analytics", label: "Chris Sale Analytics", icon: "▤" },
  { href: "/simulator", label: "Game Simulator", icon: "⚙" },
  { href: "/videos", label: "Video Library", icon: "▶" },
  { href: "/challenge", label: "Prediction Challenge", icon: "◎" },
  { href: "/performance", label: "Model Performance", icon: "▲" },
  { href: "/lab", label: "Model Lab", icon: "⚗" },
  { href: "/console", label: "Engineering Console", icon: "⌘" },
  { href: "/settings", label: "Settings", icon: "☰" },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const { pitcher, engineering, setEngineering, apiError } = useApp();
  const [open, setOpen] = useState(false);
  const nav = (
    <nav className="flex flex-col gap-0.5">
      {NAV.map((n) => {
        const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
        const label = n.href === "/analytics" && pitcher ? `${pitcher.full_name} Analytics` : n.label;
        return (
          <Link key={n.href} href={n.href} onClick={() => setOpen(false)}
            className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm ${active ? "bg-accent/10 text-accent" : "text-slate-300 hover:bg-white/5"}`}>
            <span className="w-4 text-center opacity-80">{n.icon}</span>{label}
          </Link>
        );
      })}
    </nav>
  );
  return (
    <div className="min-h-screen flex">
      <aside className="hidden lg:flex w-60 shrink-0 flex-col border-r border-edge bg-panel/60 p-3 sticky top-0 h-screen">
        <Brand />
        <div className="mt-4">{nav}</div>
        <div className="mt-auto text-[11px] text-muted px-2">Data: Baseball Savant / Statcast, MLB Stats API. Video: official links only.</div>
      </aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" onClick={() => setOpen(false)}>
          <div className="absolute inset-0 bg-black/60" />
          <aside className="absolute left-0 top-0 bottom-0 w-64 bg-panel border-r border-edge p-3" onClick={(e) => e.stopPropagation()}>
            <Brand />
            <div className="mt-4">{nav}</div>
          </aside>
        </div>
      )}
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-edge bg-bg/85 backdrop-blur px-4 py-2.5">
          <button className="lg:hidden btn-ghost px-2" onClick={() => setOpen(true)} aria-label="Open navigation">☰</button>
          <div className="lg:hidden"><Brand small /></div>
          <div className="hidden sm:block text-sm text-muted truncate">
            {pitcher ? <><span className="text-slate-200 font-medium">{pitcher.full_name}</span> · MLB ID {pitcher.pitcher_id} · throws {pitcher.throws}</> : "Loading pitcher…"}
          </div>
          <div className="ml-auto flex items-center gap-1 rounded-lg border border-edge p-0.5 text-xs">
            <button onClick={() => setEngineering(false)} className={`px-2.5 py-1 rounded-md ${!engineering ? "bg-white/10 text-white" : "text-muted"}`}>Standard</button>
            <button onClick={() => setEngineering(true)} className={`px-2.5 py-1 rounded-md ${engineering ? "bg-accent/20 text-accent" : "text-muted"}`}>Engineering</button>
          </div>
        </header>
        {apiError && <div className="m-4 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">API unreachable: {apiError}. Start the backend (see README).</div>}
        <main className="flex-1 min-w-0 p-4 md:p-6 max-w-[1600px] w-full mx-auto">{children}</main>
      </div>
    </div>
  );
}

function Brand({ small }: { small?: boolean }) {
  return (
    <Link href="/" className="flex items-center gap-2 px-2">
      <span className="h-7 w-7 rounded-lg bg-accent/15 border border-accent/40 flex items-center justify-center text-accent font-bold text-sm">IQ</span>
      {!small && <span className="font-semibold tracking-tight">PitchIQ <span className="text-accent">AI</span></span>}
    </Link>
  );
}
