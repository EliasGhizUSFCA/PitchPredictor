"use client";
import { useEffect, useRef, useState } from "react";
import { api, ApiError, useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { pct, pitchColor } from "@/lib/pitch";
import { Badge, Card, EngOnly, Empty, ErrorBox, Field, JsonInspector, Loading, Note, PageHeader, Tabs } from "@/components/ui";
import { PitchChip, ProbBars } from "@/components/pitch";
import { AttributionPanel } from "@/components/explain";
import { ProbLines } from "@/components/charts";
import { TraceTimeline } from "@/components/trace";

type Prev = { type: string; result: string; speed?: number | null; same_pa: boolean };
type SimState = {
  stand: "L" | "R"; bat_order_slot: number; times_through_order: number; balls: number; strikes: number; inning: number; outs: number;
  on_1b: boolean; on_2b: boolean; on_3b: boolean; score_diff: number; is_home: boolean; pitch_count_before: number;
  previous_pitches: Prev[]; as_of_date: string | null; days_rest: number | null;
};
const DEFAULT: SimState = {
  stand: "R", bat_order_slot: 1, times_through_order: 1, balls: 0, strikes: 0, inning: 1, outs: 0, on_1b: false, on_2b: false,
  on_3b: false, score_diff: 0, is_home: true, pitch_count_before: 0, previous_pitches: [], as_of_date: null, days_rest: 5,
};
const RESULTS = ["ball", "called_strike", "swinging_strike", "foul", "in_play", "hbp"];

function useDebounced<T>(v: T, ms = 250): T {
  const [d, setD] = useState(v);
  useEffect(() => { const t = setTimeout(() => setD(v), ms); return () => clearTimeout(t); }, [v, ms]);
  return d;
}

function useSim(pid: number | null, state: SimState) {
  const deb = useDebounced(state);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const seq = useRef(0);
  useEffect(() => {
    if (!pid) return;
    const my = ++seq.current;
    setLoading(true);
    api(`/pitchers/${pid}/simulate`, { json: clean(deb) })
      .then((r) => { if (my === seq.current) { setRes(r); setErr(null); } })
      .catch((e) => { if (my === seq.current) setErr(e); })
      .finally(() => { if (my === seq.current) setLoading(false); });
  }, [pid, deb]);
  return { res, err, loading };
}

function clean(s: SimState) {
  return { ...s, as_of_date: s.as_of_date || null, previous_pitches: s.previous_pitches.map((p) => ({ ...p, speed: p.speed || null })) };
}

export default function SimulatorPage() {
  const { pid } = useApp();
  const summary = useApi<any>(pid ? `/pitchers/${pid}/summary` : null);
  const classes: string[] = summary.data?.classes || ["FF", "SL", "CH", "SI"];
  const [state, setState] = useState<SimState>(DEFAULT);
  const [tab, setTab] = useState("predict");
  const sim = useSim(pid, state);
  return (
    <div className="space-y-5">
      <PageHeader title="Game-State Simulator" subtitle="Construct a situation; every change triggers real inference from the trained model." />
      <Note tone="warn">Simulator outputs show <b>model sensitivity, not causal effects</b>, and are kept separate from official accuracy (see Model Performance). Unspecified context uses the pitcher&apos;s history strictly before the as-of date.</Note>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
        <Card title="Scenario" right={<button className="btn-ghost text-xs" onClick={() => setState(DEFAULT)}>Reset</button>}>
          <Controls state={state} setState={setState} classes={classes} />
        </Card>
        <div className="space-y-4 min-w-0">
          <Tabs value={tab} onChange={setTab} tabs={[
            { id: "predict", label: "Prediction" }, { id: "compare", label: "Compare A vs B" }, { id: "single", label: "Single variable" },
            { id: "pitchcount", label: "Pitch-count curve" }, { id: "heatmap", label: "Count heatmap" }, { id: "saved", label: "Saved scenarios" }]} />
          {tab === "predict" && <PredictView sim={sim} />}
          {tab === "compare" && <CompareView pid={pid} a={state} classes={classes} />}
          {tab === "single" && <SweepView pid={pid} state={state} classes={classes} />}
          {tab === "pitchcount" && <SweepView pid={pid} state={state} classes={classes} fixedField="pitch_count_before" />}
          {tab === "heatmap" && <HeatmapView pid={pid} state={state} classes={classes} />}
          {tab === "saved" && <SavedView pid={pid} state={state} load={setState} />}
        </div>
      </div>
    </div>
  );
}

function Controls({ state, setState, classes }: { state: SimState; setState: (s: SimState) => void; classes: string[] }) {
  const set = <K extends keyof SimState>(k: K, v: SimState[K]) => setState({ ...state, [k]: v });
  const numIn = (k: keyof SimState, min: number, max: number) => (
    <input type="number" className="input w-full" min={min} max={max} value={state[k] as number}
      onChange={(e) => set(k, Math.max(min, Math.min(max, Number(e.target.value))) as never)} />
  );
  const seg = (k: keyof SimState, vals: (string | number | boolean)[], labels?: string[]) => (
    <div className="flex rounded-lg border border-edge overflow-hidden">
      {vals.map((v, i) => (
        <button key={String(v)} onClick={() => set(k, v as never)}
          className={`flex-1 px-2 py-1 text-xs ${state[k] === v ? "bg-accent/20 text-accent" : "text-slate-300 hover:bg-white/5"}`}>{labels?.[i] ?? String(v)}</button>
      ))}
    </div>
  );
  const prev = state.previous_pitches;
  const setPrev = (p: Prev[]) => set("previous_pitches", p);
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <Field label="Batter hand">{seg("stand", ["L", "R"], ["Left", "Right"])}</Field>
        <Field label="Lineup position">{seg("bat_order_slot", [1, 2, 3, 4, 5, 6, 7, 8, 9])}</Field>
        <Field label="Balls">{seg("balls", [0, 1, 2, 3])}</Field>
        <Field label="Strikes">{seg("strikes", [0, 1, 2])}</Field>
        <Field label="Inning">{numIn("inning", 1, 12)}</Field>
        <Field label="Outs">{seg("outs", [0, 1, 2])}</Field>
        <Field label="Times through order">{seg("times_through_order", [1, 2, 3, 4])}</Field>
        <Field label="Pitch count before">{numIn("pitch_count_before", 0, 140)}</Field>
        <Field label="Score diff (pitcher − batting)">{numIn("score_diff", -15, 15)}</Field>
        <Field label="Pitcher's team">{seg("is_home", [true, false], ["Home", "Away"])}</Field>
      </div>
      <Field label="Runners">
        <div className="flex gap-2">
          {(["on_1b", "on_2b", "on_3b"] as const).map((k, i) => (
            <button key={k} onClick={() => set(k, !state[k])} className={`flex-1 btn-ghost justify-center ${state[k] ? "!border-accent text-accent" : ""}`}>{i + 1}B</button>
          ))}
        </div>
      </Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label="History as of date">
          <input type="date" className="input" value={state.as_of_date || ""} onChange={(e) => set("as_of_date", e.target.value || null)} />
        </Field>
        <Field label="Days rest">
          <input type="number" className="input" value={state.days_rest ?? ""} onChange={(e) => set("days_rest", e.target.value === "" ? null : Number(e.target.value))} />
        </Field>
      </div>
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <span className="label">Previous pitches (oldest → newest)</span>
          <button className="btn-ghost text-xs" onClick={() => setPrev([...prev, { type: classes[0], result: "ball", same_pa: true }])}>+ Add</button>
        </div>
        {prev.length === 0 && <div className="text-xs text-muted">None entered. If pitch count &gt; 0 the model averages over the historical distribution of previous pitches for this count.</div>}
        <div className="space-y-1.5">
          {prev.map((p, i) => (
            <div key={i} className="grid grid-cols-[4.5rem_minmax(0,1fr)_4rem_auto_auto] gap-1.5 items-center">
              <select className="input px-1" value={p.type} style={{ color: pitchColor(p.type) }} onChange={(e) => setPrev(prev.map((q, j) => j === i ? { ...q, type: e.target.value } : q))}>
                {classes.map((c) => <option key={c}>{c}</option>)}
              </select>
              <select className="input px-1" value={p.result} onChange={(e) => setPrev(prev.map((q, j) => j === i ? { ...q, result: e.target.value } : q))}>
                {RESULTS.map((r) => <option key={r} value={r}>{r.replace("_", " ")}</option>)}
              </select>
              <input className="input px-1" placeholder="mph" type="number" value={p.speed ?? ""} onChange={(e) => setPrev(prev.map((q, j) => j === i ? { ...q, speed: e.target.value ? Number(e.target.value) : null } : q))} />
              <label className="flex items-center gap-1 text-[11px] text-muted"><input type="checkbox" checked={p.same_pa} onChange={(e) => setPrev(prev.map((q, j) => j === i ? { ...q, same_pa: e.target.checked } : q))} />PA</label>
              <button className="text-muted hover:text-red-300 px-1" onClick={() => setPrev(prev.filter((_, j) => j !== i))} aria-label="remove">✕</button>
            </div>
          ))}
        </div>
        <div className="text-[10px] text-muted mt-1">“PA” = thrown in the current plate appearance; their results must produce the selected count.</div>
      </div>
    </div>
  );
}

function PredictView({ sim }: { sim: { res: any; err: ApiError | null; loading: boolean } }) {
  const { res, err, loading } = sim;
  if (err) return <ErrorBox error={err} what={err.status === 422 ? "Invalid game state" : "Inference failed"} />;
  if (!res) return <Loading label="Running inference…" />;
  const top = res.predictions[0];
  const sup = res.support;
  return (
    <div className={`space-y-4 transition-opacity ${loading ? "opacity-60" : ""}`}>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Predicted next pitch" subtitle={`Model ${res.model_family} · ${res.calibration_status}`}
          right={<span className="text-[11px] text-muted tnum">{res.trace.total_ms.toFixed(0)} ms</span>}>
          <div className="flex items-center gap-2 mb-3"><PitchChip code={top.pitch} size="lg" /><span className="tnum text-lg font-semibold">{pct(top.probability)}</span></div>
          <ProbBars predictions={res.predictions} />
        </Card>
        <Card title="Historical support" subtitle="How much training data resembles this scenario">
          <div className="grid grid-cols-2 gap-2 text-sm tnum">
            <div>Same hand & count: <b>{sup.same_hand_and_count}</b></div>
            <div>Pitch count ±5: <b>{sup.pitch_count_within_5}</b></div>
            <div>Same inning: <b>{sup.same_inning}</b></div>
            <div>Similar situations: <b>{sup.similar_situations}</b></div>
          </div>
          <div className="text-[11px] text-muted mt-1">of {sup.training_rows} training pitches</div>
          {sup.out_of_distribution ? (
            <div className="mt-3 space-y-1">{sup.warnings.map((w: string) => <div key={w} className="text-xs text-amber-300">⚠ {w}</div>)}</div>
          ) : <div className="mt-3"><Badge tone="ok">within training distribution</Badge></div>}
          {(res.assumptions?.imputed?.length > 0 || res.assumptions?.marginalized_over) && (
            <div className="mt-3 text-xs text-cyan-100/80 space-y-1">
              {res.assumptions.imputed.map((x: string) => <div key={x}>• Imputed: {x}</div>)}
              {res.assumptions.marginalized_over && <div>• Previous pitch unknown — {res.assumptions.marginalized_over.method} ({res.assumptions.marginalized_over.combinations.length} combinations).</div>}
            </div>
          )}
          <div className="text-[11px] text-muted mt-2">History as of {res.assumptions?.as_of_date}</div>
        </Card>
      </div>
      <Card title="Feature contributions"><AttributionPanel attribution={res.attribution} input={res.input_features} /></Card>
      <EngOnly>
        <Card title="Pipeline trace"><TraceTimeline trace={res.trace} /><div className="mt-2"><JsonInspector data={res.input_features} label="input_features" /></div></Card>
      </EngOnly>
    </div>
  );
}

function CompareView({ pid, a, classes }: { pid: number | null; a: SimState; classes: string[] }) {
  const [b, setB] = useState<SimState>({ ...a, balls: 0, strikes: 2, inning: 7, pitch_count_before: 95, times_through_order: 3, previous_pitches: [] });
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const da = useDebounced(a), db = useDebounced(b);
  useEffect(() => {
    if (!pid) return;
    api(`/pitchers/${pid}/simulate/compare`, { json: { a: clean(da), b: clean(db) } }).then((r) => { setRes(r); setErr(null); }).catch(setErr);
  }, [pid, da, db]);
  return (
    <div className="space-y-4">
      <div className="text-xs text-muted">Scenario A = the main scenario panel. Edit Scenario B here.</div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Scenario B"><Controls state={b} setState={setB} classes={classes} /></Card>
        <div className="space-y-4 min-w-0">
          <ErrorBox error={err} />
          {res && (
            <>
              <Card title="Probabilities" right={res.top_changed ? <Badge tone="warning">top pitch changes</Badge> : <Badge>same top pitch</Badge>}>
                <div className="grid grid-cols-2 gap-4">
                  <div><div className="label mb-2">A</div><ProbBars predictions={res.a.predictions} compact /></div>
                  <div><div className="label mb-2">B</div><ProbBars predictions={res.b.predictions} compact /></div>
                </div>
              </Card>
              <Card title="B − A (percentage points)">
                <div className="space-y-1">{Object.entries(res.percentage_point_diff as Record<string, number>).map(([c, v]) => (
                  <div key={c} className="flex items-center gap-2 text-sm"><PitchChip code={c} /><span className={`tnum ${v >= 0 ? "text-cyan-300" : "text-red-300"}`}>{v >= 0 ? "+" : ""}{v.toFixed(1)} pp</span></div>
                ))}</div>
                <div className="grid grid-cols-2 gap-2 text-xs text-muted mt-3 tnum">
                  <div>A support: {res.a.support.similar_situations} similar{res.a.support.out_of_distribution ? " ⚠ OOD" : ""}</div>
                  <div>B support: {res.b.support.similar_situations} similar{res.b.support.out_of_distribution ? " ⚠ OOD" : ""}</div>
                </div>
              </Card>
              {res.attribution_diff && (
                <Card title={`Attribution difference toward ${res.attribution_diff.class} (B − A, by group)`}>
                  {Object.entries(res.attribution_diff.by_group as Record<string, number>).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1])).map(([g, v]) => (
                    <div key={g} className="flex justify-between text-xs py-0.5"><span>{g}</span><span className={`tnum ${v >= 0 ? "text-cyan-300" : "text-red-300"}`}>{v >= 0 ? "+" : ""}{v.toFixed(3)}</span></div>
                  ))}
                </Card>
              )}
              <div className="text-[11px] text-muted">{res.note}</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

const SWEEP_FIELDS = [
  { id: "stand", label: "Batter handedness" }, { id: "count", label: "Ball-strike count" }, { id: "inning", label: "Inning" },
  { id: "pitch_count_before", label: "Pitch count" }, { id: "bat_order_slot", label: "Lineup position" },
  { id: "times_through_order", label: "Times through order" }, { id: "outs", label: "Outs" }, { id: "score_diff", label: "Score differential" },
];

function SweepView({ pid, state, classes, fixedField }: { pid: number | null; state: SimState; classes: string[]; fixedField?: string }) {
  const [field, setField] = useState(fixedField || "stand");
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const ds = useDebounced(state, 400);
  const f = fixedField || field;
  useEffect(() => {
    if (!pid) return;
    api(`/pitchers/${pid}/simulate/sweep`, { json: { state: clean(ds), field: f } }).then((r) => { setRes(r); setErr(null); }).catch(setErr);
  }, [pid, ds, f]);
  return (
    <Card title={fixedField ? "Pitch-count sensitivity" : "Single-variable analysis"}
      subtitle="One input varied, everything else held at the scenario values. Invalid combinations (e.g. TTO vs pitch count) are skipped."
      right={!fixedField && <select className="input" value={field} onChange={(e) => setField(e.target.value)}>{SWEEP_FIELDS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}</select>}>
      <ErrorBox error={err} />
      {!res && !err && <Loading />}
      {res && res.points.length === 0 && <Empty>No valid values for this scenario.</Empty>}
      {res && res.points.length > 0 && (
        <>
          {res.points.length <= 4 ? (
            <div className="grid gap-4 sm:grid-cols-2">
              {res.points.map((p: any) => (
                <div key={String(p.value)} className="card p-3 bg-panel2/40"><div className="label mb-2">{f} = {String(p.value)}</div>
                  <ProbBars compact predictions={Object.entries(p.probabilities as Record<string, number>).map(([pitch, probability]) => ({ pitch, probability })).sort((a, b) => b.probability - a.probability)} />
                </div>
              ))}
            </div>
          ) : <ProbLines points={res.points} classes={res.classes || classes} xLabel={f} />}
          <div className="text-[11px] text-muted mt-2 tnum">{res.points.length} model evaluations · {res.inference_ms.toFixed(0)} ms · {res.note}</div>
        </>
      )}
    </Card>
  );
}

function HeatmapView({ pid, state, classes }: { pid: number | null; state: SimState; classes: string[] }) {
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [sel, setSel] = useState<string>("top");
  const ds = useDebounced(state, 400);
  useEffect(() => {
    if (!pid) return;
    api(`/pitchers/${pid}/simulate/sweep`, { json: { state: clean(ds), field: "count" } }).then((r) => { setRes(r); setErr(null); }).catch(setErr);
  }, [pid, ds]);
  const byCount: Record<string, Record<string, number>> = {};
  res?.points.forEach((p: any) => (byCount[p.value] = p.probabilities));
  return (
    <Card title="12-count heatmap" subtitle="Model prediction in each count (same-PA sequence removed; other inputs held constant)"
      right={<select className="input" value={sel} onChange={(e) => setSel(e.target.value)}>
        <option value="top">Highest-probability pitch</option>{classes.map((c) => <option key={c} value={c}>P({c})</option>)}</select>}>
      <ErrorBox error={err} />
      {!res && !err && <Loading />}
      {res && (
        <div className="max-w-lg">
          <div className="grid gap-1.5" style={{ gridTemplateColumns: "4rem repeat(3, minmax(0,1fr))" }}>
            <div />{[0, 1, 2].map((s) => <div key={s} className="text-center label">{s} strikes</div>)}
            {[0, 1, 2, 3].map((b) => [
              <div key={`b${b}`} className="label flex items-center">{b} balls</div>,
              ...[0, 1, 2].map((s) => {
                const p = byCount[`${b}-${s}`];
                if (!p) return <div key={`${b}${s}`} className="rounded-md border border-edge h-16 flex items-center justify-center text-xs text-muted">—</div>;
                const top = Object.entries(p).sort((x, y) => y[1] - x[1])[0];
                const code = sel === "top" ? top[0] : sel;
                const v = sel === "top" ? top[1] : p[sel];
                const c = pitchColor(code);
                return (
                  <div key={`${b}${s}`} className="rounded-md border border-edge h-16 flex flex-col items-center justify-center"
                    style={{ background: `${c}${Math.round(Math.min(1, v) * 200).toString(16).padStart(2, "0")}` }}>
                    <div className="text-xs font-semibold">{b}-{s}</div>
                    <div className="text-sm font-semibold tnum">{sel === "top" ? `${code} ${pct(v, 0)}` : pct(v, 0)}</div>
                  </div>
                );
              }),
            ])}
          </div>
        </div>
      )}
    </Card>
  );
}

function SavedView({ pid, state, load }: { pid: number | null; state: SimState; load: (s: SimState) => void }) {
  const list = useApi<any[]>(pid ? `/pitchers/${pid}/scenarios` : null);
  const [name, setName] = useState("");
  const [err, setErr] = useState<ApiError | null>(null);
  async function save() {
    try { await api(`/pitchers/${pid}/scenarios`, { json: { name: name || `Scenario ${new Date().toLocaleString()}`, state: clean(state) } }); setName(""); list.reload(); }
    catch (e) { setErr(e as ApiError); }
  }
  async function del(id: number) { await api(`/pitchers/${pid}/scenarios/${id}`, { method: "DELETE" }).catch(setErr); list.reload(); }
  function exportJson(obj: unknown, fname: string) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = fname; a.click(); URL.revokeObjectURL(url);
  }
  return (
    <Card title="Saved scenarios" subtitle="Stored in PostgreSQL; simulator scenarios never feed official accuracy metrics">
      <div className="flex flex-wrap gap-2 mb-3">
        <input className="input flex-1 min-w-[10rem]" placeholder="Scenario name" value={name} onChange={(e) => setName(e.target.value)} />
        <button className="btn-primary" onClick={save} disabled={!pid}>Save current</button>
        <button className="btn-ghost" onClick={() => exportJson(clean(state), "pitchiq-scenario.json")}>Export current</button>
        {list.data && list.data.length > 0 && <button className="btn-ghost" onClick={() => exportJson(list.data, "pitchiq-scenarios.json")}>Export all</button>}
      </div>
      <ErrorBox error={err || list.error} />
      {list.data?.length === 0 && <Empty>No saved scenarios.</Empty>}
      <div className="space-y-1.5">
        {list.data?.map((s) => (
          <div key={s.scenario_id} className="flex flex-wrap items-center gap-2 rounded-md border border-edge bg-panel2 px-3 py-2 text-sm">
            <span className="font-medium">{s.name}</span>
            <span className="text-xs text-muted tnum">{s.state.stand}HB · {s.state.balls}-{s.state.strikes} · inn {s.state.inning} · {s.state.pitch_count_before} pitches</span>
            <span className="ml-auto flex gap-1.5">
              <button className="btn-ghost text-xs" onClick={() => load({ ...DEFAULT, ...s.state })}>Open</button>
              <button className="btn-danger text-xs" onClick={() => del(s.scenario_id)}>Delete</button>
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}
