"use client";
import { createContext, useContext, useEffect, useState } from "react";
import { api } from "./api";

type Pitcher = { pitcher_id: number; full_name: string; throws: string; seasons: number[]; pitches: number; is_active_target: boolean };
type Ctx = {
  pitchers: Pitcher[];
  pid: number | null;
  pitcher: Pitcher | null;
  setPid: (p: number) => void;
  engineering: boolean;
  setEngineering: (b: boolean) => void;
  apiError: string | null;
};
const AppCtx = createContext<Ctx>({
  pitchers: [], pid: null, pitcher: null, setPid: () => {}, engineering: false, setEngineering: () => {}, apiError: null,
});

function read(key: string): string | null {
  try { return window.localStorage.getItem(key); } catch { return null; }
}
function write(key: string, v: string) {
  try { window.localStorage.setItem(key, v); } catch { /* storage unavailable */ }
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [pitchers, setPitchers] = useState<Pitcher[]>([]);
  const [pid, setPidState] = useState<number | null>(null);
  const [engineering, setEng] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  useEffect(() => {
    setEng(read("pitchiq.view") === "engineering");
    api<Pitcher[]>("/pitchers")
      .then((ps) => {
        setPitchers(ps);
        const saved = Number(read("pitchiq.pid"));
        const pick = ps.find((p) => p.pitcher_id === saved) || ps.find((p) => p.is_active_target) || ps[0];
        if (pick) setPidState(pick.pitcher_id);
      })
      .catch((e) => setApiError(String(e.message || e)));
  }, []);
  const setPid = (p: number) => { setPidState(p); write("pitchiq.pid", String(p)); };
  const setEngineering = (b: boolean) => { setEng(b); write("pitchiq.view", b ? "engineering" : "standard"); };
  const pitcher = pitchers.find((p) => p.pitcher_id === pid) || null;
  return (
    <AppCtx.Provider value={{ pitchers, pid, pitcher, setPid, engineering, setEngineering, apiError }}>
      {children}
    </AppCtx.Provider>
  );
}
export const useApp = () => useContext(AppCtx);
