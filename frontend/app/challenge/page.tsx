"use client";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError, useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { ci, num, pct } from "@/lib/pitch";
import { Badge, Card, EngOnly, Empty, ErrorBox, Field, Loading, Note, PageHeader, Stat, Table } from "@/components/ui";
import { PitchChip, ProbBars, Pred } from "@/components/pitch";
import { SituationPanel } from "@/components/situation";
import { AttributionPanel } from "@/components/explain";
import { LockedVideo, RevealDetails, VideoReveal } from "@/components/reveal";
import { MatrixGrid, Reliability, axis, tooltipStyle } from "@/components/charts";
import { TraceTimeline } from "@/components/trace";

const STRATS = [
  { id: "natural", label: "Natural random sampling" }, { id: "balanced_pitch_type", label: "Balanced by pitch type" },
  { id: "balanced_count", label: "Balanced by ball-strike count" }, { id: "lhb", label: "Left-handed batters" },
  { id: "rhb", label: "Right-handed batters" }, { id: "innings", label: "Specific innings" }, { id: "pitch_count_range", label: "Pitch-count range" },
];

export default function ChallengePage() {
  const { pid } = useApp();
  const [size, setSize] = useState(25);
  const [custom, setCustom] = useState("");
  const [strategy, setStrategy] = useState("natural");
  const [innings, setInnings] = useState("1,2,3");
  const [pcMin, setPcMin] = useState(0);
  const [pcMax, setPcMax] = useState(50);
  const [seed, setSeed] = useState("");
  const [challenge, setChallenge] = useState<any>(null);
  const [att, setAtt] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);
  const past = useApi<any[]>(pid ? `/pitchers/${pid}/challenges` : null);
  const summary = useApi<any>(pid ? `/pitchers/${pid}/summary` : null);
  const classes: string[] = summary.data?.classes || [];

  async function run<T>(fn: () => Promise<T>) {
    setBusy(true); setErr(null);
    try { return await fn(); } catch (e) { setErr(e as ApiError); } finally { setBusy(false); }
  }
  async function start() {
    const n = custom ? Number(custom) : size;
    const params = strategy === "innings" ? { innings: innings.split(",").map((x) => Number(x.trim())).filter(Boolean) }
      : strategy === "pitch_count_range" ? { min: pcMin, max: pcMax } : undefined;
    await run(async () => {
      const c = await api(`/pitchers/${pid}/challenges`, { json: { size: n, strategy, params, seed: seed ? Number(seed) : null } });
      setChallenge(c); setStats(null); past.reload();
      await next(c.challenge_id);
    });
  }
  async function resume(c: any) {
    await run(async () => { setChallenge(c); await next(c.challenge_id); setStats(await api(`/challenges/${c.challenge_id}/stats`)); });
  }
  async function next(cid = challenge?.challenge_id) {
    const a = await api(`/challenges/${cid}/next`, { method: "POST" });
    setAtt(a);
  }
  async function guess(g: string) {
    await run(async () => { await api(`/attempts/${att.attempt_id}/guess`, { json: { guess: g } }); setAtt({ ...att, user_guess: g }); });
  }
  async function reveal() {
    await run(async () => {
      const r = await api(`/attempts/${att.attempt_id}/reveal`, { method: "POST" });
      setAtt({ ...r, trace: att.trace });
      setStats(await api(`/challenges/${challenge.challenge_id}/stats`));
    });
  }
  const balanced = strategy.startsWith("balanced");
  return (
    <div className="space-y-5">
      <PageHeader title="PitchIQ Prediction Challenge" subtitle="Blind test on random pitches from the locked held-out test split. The prediction is frozen before the answer is loaded." />
      {!challenge && (
        <Card title="New challenge">
          <div className="grid gap-3 md:grid-cols-4">
            <Field label="Pitches">
              <div className="flex gap-1 flex-wrap">
                {[10, 25, 50, 100].map((n) => <button key={n} onClick={() => { setSize(n); setCustom(""); }} className={`btn-ghost ${size === n && !custom ? "!border-accent text-accent" : ""}`}>{n}</button>)}
                <input className="input w-20" placeholder="custom" value={custom} onChange={(e) => setCustom(e.target.value.replace(/\D/g, ""))} />
              </div>
            </Field>
            <Field label="Sampling strategy"><select className="input" value={strategy} onChange={(e) => setStrategy(e.target.value)}>{STRATS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}</select></Field>
            {strategy === "innings" && <Field label="Innings (comma list)"><input className="input" value={innings} onChange={(e) => setInnings(e.target.value)} /></Field>}
            {strategy === "pitch_count_range" && <Field label="Pitch count range"><div className="flex gap-1"><input type="number" className="input w-20" value={pcMin} onChange={(e) => setPcMin(Number(e.target.value))} /><input type="number" className="input w-20" value={pcMax} onChange={(e) => setPcMax(Number(e.target.value))} /></div></Field>}
            <Field label="Seed (optional)"><input className="input" placeholder="random" value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} /></Field>
          </div>
          {balanced && <div className="mt-3"><Note tone="warn">Balanced sampling over-represents rare pitches/counts. Results from this session do <b>not</b> reflect natural pitch frequencies or official accuracy.</Note></div>}
          <div className="mt-4 flex gap-2 items-center"><button className="btn-primary" onClick={start} disabled={busy || !pid}>Start challenge</button><ErrorBox error={err} /></div>
        </Card>
      )}
      {challenge && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
          <Badge>challenge #{challenge.challenge_id}</Badge><Badge>{challenge.strategy ?? challenge.sampling_strategy}</Badge><span>seed {challenge.seed}</span>
          <span className="font-mono truncate">{challenge.model_version}</span>
          <button className="btn-ghost ml-auto" onClick={() => { setChallenge(null); setAtt(null); setStats(null); }}>New challenge</button>
        </div>
      )}
      {challenge && <ErrorBox error={err} />}
      {challenge && att?.done && <Card><div className="text-sm">Challenge complete — {stats?.tested ?? 0} pitches tested. See results below.</div></Card>}
      {challenge && att && !att.done && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,18rem)_minmax(0,1fr)_minmax(0,20rem)]">
          <Card title={`Pitch ${att.position + 1} of ${att.total}`} className="order-2 xl:order-1"><SituationPanel s={att.situation} /></Card>
          <div className="order-1 xl:order-2 space-y-3">
            {att.revealed ? <VideoReveal v={att.video_reference} /> : <LockedVideo mode={att.mode} available={att.video.available} />}
            {att.revealed ? (
              <>
                <RevealDetails r={att} />
                <button className="btn-primary" onClick={() => run(() => next())} disabled={busy}>Next pitch →</button>
              </>
            ) : (
              <div className="card p-3 space-y-3">
                <div className="text-xs text-muted">Your guess (optional, scored separately from the model):</div>
                <div className="flex flex-wrap gap-1.5">
                  {att.prediction.predictions.map((p: Pred) => (
                    <button key={p.pitch} onClick={() => guess(p.pitch)} className={`btn-ghost ${att.user_guess === p.pitch ? "!border-accent" : ""}`}><PitchChip code={p.pitch} /></button>
                  ))}
                </div>
                <button className="btn-primary" onClick={reveal} disabled={busy}>Reveal pitch</button>
              </div>
            )}
          </div>
          <Card title="Frozen model prediction" className="order-3" subtitle={`prediction #${att.prediction.prediction_id} · saved ${new Date(att.prediction.frozen_at).toLocaleTimeString()}`}>
            <ProbBars predictions={att.prediction.predictions} actual={att.revealed ? att.actual?.model_class : null} />
            <div className="text-[11px] text-muted mt-3 tnum">inference {num(att.prediction.inference_ms, 1)} ms · video mode: {att.mode}</div>
          </Card>
        </div>
      )}
      {challenge && att && !att.done && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Feature contributions"><AttributionPanel attribution={att.prediction.attribution} /></Card>
          <EngOnly><Card title="Processing diagnostics">{att.trace ? <TraceTimeline trace={att.trace} /> : <div className="text-sm text-muted">Trace {att.prediction.trace_id} — open in Engineering Console.</div>}</Card></EngOnly>
        </div>
      )}
      {challenge && <StatsPanel stats={stats} classes={classes} />}
      <Card title="Past challenges">
        <ErrorBox error={past.error} />
        {past.data?.length === 0 && <Empty>None yet.</Empty>}
        {past.data && past.data.length > 0 && (
          <Table>
            <thead><tr><th className="th">#</th><th className="th">Created</th><th className="th">Size</th><th className="th">Strategy</th><th className="th">Seed</th><th className="th">Status</th><th className="th" /></tr></thead>
            <tbody>{past.data.map((c) => (
              <tr key={c.challenge_id}><td className="td">{c.challenge_id}</td><td className="td">{new Date(c.created_at).toLocaleString()}</td><td className="td">{c.size}</td>
                <td className="td">{c.strategy}</td><td className="td">{c.seed}</td><td className="td"><Badge tone={c.status === "completed" ? "completed" : "running"}>{c.status}</Badge></td>
                <td className="td"><button className="btn-ghost text-xs" onClick={() => resume(c)}>Open</button></td></tr>))}</tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}

function StatsPanel({ stats, classes }: { stats: any; classes: string[] }) {
  if (!stats) return <Card title="Live accuracy"><Empty>Reveal a pitch to start scoring.</Empty></Card>;
  if (!stats.tested) return <Card title="Live accuracy"><Empty>No revealed pitches yet.</Empty></Card>;
  const cls = stats.classes || classes;
  return (
    <Card title="Live accuracy" subtitle={stats.note}
      right={!stats.natural_frequencies ? <Badge tone="warning">balanced sample — not natural frequencies</Badge> : <Badge>natural sample</Badge>}>
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-3 mb-4">
        <Stat label="Pitches tested" value={`${stats.tested} / ${stats.size}`} />
        <Stat label="Correct top-1" value={stats.top1_correct} />
        <Stat label="Top-1 accuracy" accent value={pct(stats.top1_accuracy)} sub={`95% CI ${ci(stats.top1_ci95)}`} />
        <Stat label="Top-2 accuracy" value={pct(stats.top2_accuracy)} />
        <Stat label="Log loss" value={num(stats.log_loss)} />
        <Stat label="Brier" value={num(stats.brier)} />
        <Stat label="Mean P(actual)" value={pct(stats.mean_prob_actual)} />
      </div>
      <div className="grid gap-4 xl:grid-cols-3">
        <div>
          <div className="label mb-2">Cumulative accuracy</div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={stats.cumulative_accuracy.map((v: number, i: number) => ({ i: i + 1, v }))} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
              <CartesianGrid stroke="#1e2633" strokeDasharray="3 3" /><XAxis dataKey="i" {...axis} /><YAxis domain={[0, 1]} {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
              <Tooltip {...tooltipStyle} formatter={(v: number) => pct(v)} /><Line dataKey="v" stroke="#22d3ee" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="label mb-2">Confusion matrix (actual × predicted)</div>
          <MatrixGrid rows={stats.confusion.labels} cols={stats.confusion.labels} matrix={stats.confusion.matrix} rowTitle="actual" colTitle="predicted" />
        </div>
        <div>
          <div className="label mb-2">Reliability (top-label)</div>
          <Reliability bins={stats.calibration.top_label_bins} height={200} />
          <div className="text-[11px] text-muted tnum">ECE {num(stats.calibration.ece_top_label)} (small samples are noisy)</div>
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-2 mt-4">
        <div>
          <div className="label mb-2">Accuracy by actual pitch type</div>
          <Table><thead><tr><th className="th">Pitch</th><th className="th">n</th><th className="th">Correct</th><th className="th">Accuracy</th><th className="th">95% CI</th></tr></thead>
            <tbody>{cls.filter((c: string) => stats.per_pitch_type[c]).map((c: string) => { const r = stats.per_pitch_type[c]; return (
              <tr key={c}><td className="td"><PitchChip code={c} /></td><td className="td">{r.n}</td><td className="td">{r.correct}</td><td className="td">{pct(r.accuracy)}</td><td className="td text-muted">{ci(r.ci95)}</td></tr>); })}</tbody></Table>
        </div>
        <div>
          <div className="label mb-2">Same pitches: model vs baselines vs you</div>
          <Table><thead><tr><th className="th">Predictor</th><th className="th">Top-1</th><th className="th">Log loss</th><th className="th">Brier</th></tr></thead>
            <tbody>
              <tr><td className="td">Challenge model</td><td className="td">{pct(stats.top1_accuracy)}</td><td className="td">{num(stats.log_loss)}</td><td className="td">{num(stats.brier)}</td></tr>
              {Object.entries(stats.baselines_same_pitches || {}).map(([k, b]: [string, any]) => (
                <tr key={k}><td className="td">{k} baseline</td><td className="td">{pct(b.top1_accuracy)}</td><td className="td">{num(b.log_loss)}</td><td className="td">{num(b.brier)}</td></tr>))}
              <tr><td className="td">You ({stats.user.guesses} guesses)</td><td className="td">{stats.user.accuracy == null ? "—" : pct(stats.user.accuracy)}</td><td className="td text-muted">n/a</td><td className="td text-muted">n/a</td></tr>
            </tbody></Table>
        </div>
      </div>
      <EngOnly>
        <div className="mt-4"><div className="label mb-2">Attempt log</div>
          <Table><thead><tr><th className="th">#</th><th className="th">Pitch</th><th className="th">Top</th><th className="th">Actual</th><th className="th">P(actual)</th><th className="th">Guess</th><th className="th">Mode</th></tr></thead>
            <tbody>{stats.attempts.map((a: any) => (<tr key={a.position}><td className="td">{a.position + 1}</td><td className="td">{a.pitch_uid}</td><td className="td"><PitchChip code={a.top_pitch} /></td>
              <td className="td"><PitchChip code={a.actual} /></td><td className="td">{pct(a.prob_actual)}</td><td className="td">{a.user_guess || "—"}</td><td className="td">{a.mode}</td></tr>))}</tbody></Table>
        </div>
      </EngOnly>
    </Card>
  );
}
