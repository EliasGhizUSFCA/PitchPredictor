"use client";
import { useEffect, useState } from "react";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis, ReferenceLine } from "recharts";
import { useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { ci, num, pct, pitchColor } from "@/lib/pitch";
import { Badge, Card, EngOnly, Empty, ErrorBox, JsonInspector, Loading, Note, PageHeader, Table, Tabs } from "@/components/ui";
import { PitchChip } from "@/components/pitch";
import { BreakdownChart, MatrixGrid, Reliability, axis, grid, tooltipStyle } from "@/components/charts";

const BREAKDOWNS = ["count", "batter_hand", "batting_order", "inning", "pitch_count", "times_through_order", "season", "actual_pitch", "opponent"];

export default function PerformancePage() {
  const { pid } = useApp();
  const lb = useApi<any>(pid ? `/pitchers/${pid}/leaderboard` : null);
  const evals = useApi<any[]>(pid ? `/pitchers/${pid}/evaluations` : null);
  const [runId, setRunId] = useState<number | null>(null);
  useEffect(() => {
    if (!runId && lb.data?.rows?.length) {
      const promoted = lb.data.rows.find((r: any) => r.status === "promoted") || lb.data.rows[0];
      setRunId(promoted.evaluation_run_id);
    }
  }, [lb.data, runId]);
  const run = useApi<any>(runId ? `/evaluations/${runId}` : null);
  const [bd, setBd] = useState("count");
  const ablations = (evals.data || []).filter((e) => e.kind === "ablation");
  const walk = (evals.data || []).filter((e) => e.kind === "walk_forward");
  const [abl, setAbl] = useState<number | null>(null);
  const ablRun = useApi<any>(abl || ablations[0] ? `/evaluations/${abl || ablations[0].evaluation_run_id}` : null, [abl, ablations[0]?.evaluation_run_id]);
  const m = run.data?.metrics;
  return (
    <div className="space-y-5">
      <PageHeader title="Model Performance" subtitle="Official metrics from full benchmarks over every eligible pitch in the locked held-out split (2026 season)." />
      <Card title="Leaderboard — held-out test split" subtitle={lb.data?.note}
        right={lb.data && <Badge tone={lb.data.same_examples ? "ok" : "warning"}>{lb.data.same_examples ? "all models on the same examples" : "sample sizes differ"}</Badge>}>
        <ErrorBox error={lb.error} />
        {lb.loading && !lb.data && <Loading />}
        {lb.data?.rows?.length === 0 && <Empty>No benchmarks yet — run the benchmark job from the Engineering Console.</Empty>}
        {lb.data?.rows?.length > 0 && (
          <Table>
            <thead><tr><th className="th">Model</th><th className="th">Status</th><th className="th">n</th><th className="th">Top-1</th><th className="th">Top-1 CI (game bootstrap)</th><th className="th">Top-2</th>
              <th className="th">Log loss</th><th className="th">LL CI</th><th className="th">Brier</th><th className="th">ECE</th><th className="th">Raw LL (uncal.)</th><th className="th">ms / pitch</th><th className="th">Val LL</th></tr></thead>
            <tbody>{lb.data.rows.map((r: any) => (
              <tr key={r.model_version} className={`cursor-pointer hover:bg-white/5 ${r.evaluation_run_id === runId ? "bg-accent/5" : ""}`} onClick={() => setRunId(r.evaluation_run_id)}>
                <td className="td"><div className="font-medium">{r.family}</div><div className="text-[10px] text-muted font-mono">{r.model_version}</div></td>
                <td className="td"><Badge tone={r.status}>{r.status}</Badge></td><td className="td">{r.n}</td>
                <td className="td font-semibold">{pct(r.top1_accuracy)}</td><td className="td text-muted">{ci(r.bootstrap?.top1_accuracy_ci95)}</td>
                <td className="td">{pct(r.top2_accuracy)}</td><td className="td font-semibold">{num(r.log_loss)}</td>
                <td className="td text-muted">{r.bootstrap ? `${num(r.bootstrap.log_loss_ci95[0])} – ${num(r.bootstrap.log_loss_ci95[1])}` : "—"}</td>
                <td className="td">{num(r.brier)}</td><td className="td">{num(r.ece)}</td><td className="td text-muted">{num(r.uncalibrated?.log_loss)}</td>
                <td className="td">{r.inference_ms_per_pitch ? r.inference_ms_per_pitch.toFixed(3) : "—"}</td><td className="td text-muted">{num(r.validation?.log_loss)}</td>
              </tr>))}</tbody>
          </Table>
        )}
        <div className="mt-3"><Note>Promotion is decided on validation (2025) log loss, never on test results. Overlapping game-bootstrap intervals mean the ranking between those models is not statistically reliable.</Note></div>
      </Card>

      <Card title="Evaluation run detail" right={
        <select className="input max-w-[22rem]" value={runId ?? ""} onChange={(e) => setRunId(Number(e.target.value))}>
          {(evals.data || []).filter((e) => e.kind !== "ablation").map((e) => <option key={e.evaluation_run_id} value={e.evaluation_run_id}>
            #{e.evaluation_run_id} · {e.kind} · {e.family} · {e.split} · n={e.n_samples}{e.model_status === "archived" ? " (archived)" : ""}</option>)}
        </select>}>
        <ErrorBox error={run.error} />
        {run.loading && !m && <Loading />}
        {m && (
          <div className="space-y-5">
            <div className="flex flex-wrap gap-2 text-xs text-muted"><Badge>{run.data.kind}</Badge><Badge>{run.data.split}</Badge><span className="font-mono">{run.data.model_version}</span><span>· {new Date(run.data.created_at).toLocaleString()}</span></div>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-sm tnum">
              {[["n", m.summary.n], ["Top-1", pct(m.summary.top1_accuracy)], ["Top-2", pct(m.summary.top2_accuracy)], ["Log loss", num(m.summary.log_loss)], ["Brier", num(m.summary.brier)], ["Mean P(actual)", pct(m.summary.mean_prob_actual)]].map(([k, v]) => (
                <div key={String(k)} className="bg-panel2 border border-edge rounded-md px-2 py-1.5"><div className="label">{k}</div><div className="font-semibold">{v}</div></div>))}
            </div>
            <div className="grid gap-5 xl:grid-cols-2">
              <div>
                <div className="label mb-2">Confusion matrix</div>
                <MatrixGrid rows={m.confusion.labels} cols={m.confusion.labels} matrix={m.confusion.matrix} rowTitle="actual" colTitle="predicted" />
                <div className="mt-3"><Table><thead><tr><th className="th">Class</th><th className="th">Support</th><th className="th">Predicted</th><th className="th">Precision</th><th className="th">Recall</th><th className="th">F1</th></tr></thead>
                  <tbody>{m.confusion.labels.map((c: string) => { const p = m.confusion.per_class[c]; return (
                    <tr key={c}><td className="td"><PitchChip code={c} /></td><td className="td">{p.support}</td><td className="td">{p.predicted}</td><td className="td">{pct(p.precision)}</td><td className="td">{pct(p.recall)}</td><td className="td">{num(p.f1)}</td></tr>); })}</tbody></Table></div>
              </div>
              <div>
                <div className="label mb-2">Reliability diagram (top-label + per class one-vs-rest)</div>
                <Reliability bins={m.calibration.top_label_bins} perClass={m.calibration.per_class} height={300} />
                <div className="text-[11px] text-muted tnum">Top-label ECE {num(m.calibration.ece_top_label)}. Dashed diagonal = perfect calibration.</div>
                <div className="label mt-4 mb-2">Actual distribution vs mean predicted probability</div>
                <Table><thead><tr><th className="th">Class</th><th className="th">Actual n</th><th className="th">Actual share</th><th className="th">Mean predicted</th><th className="th">Times top pick</th></tr></thead>
                  <tbody>{Object.keys(m.pitch_distribution).map((c) => (
                    <tr key={c}><td className="td"><PitchChip code={c} /></td><td className="td">{m.pitch_distribution[c]}</td><td className="td">{pct(m.pitch_distribution[c] / m.summary.n)}</td>
                      <td className="td">{pct(m.mean_predicted_probability?.[c])}</td><td className="td">{m.predicted_distribution?.[c]}</td></tr>))}</tbody></Table>
              </div>
            </div>
            <div>
              <Tabs value={bd} onChange={setBd} tabs={BREAKDOWNS.filter((b) => m.breakdowns[b]).map((b) => ({ id: b, label: b.replaceAll("_", " ") }))} />
              <BreakdownChart rows={m.breakdowns[bd]} height={240} />
              <div className="max-h-72 overflow-auto mt-2">
                <Table><thead><tr><th className="th">{bd}</th><th className="th">n</th><th className="th">Top-1</th><th className="th">95% CI (Wilson)</th><th className="th">Log loss</th><th className="th">Brier</th></tr></thead>
                  <tbody>{m.breakdowns[bd].map((r: any) => (<tr key={r.value}><td className="td">{r.value}</td><td className="td">{r.n}</td><td className="td">{pct(r.top1_accuracy)}</td><td className="td text-muted">{ci(r.ci95)}</td><td className="td">{num(r.log_loss)}</td><td className="td">{num(r.brier)}</td></tr>))}</tbody></Table>
              </div>
            </div>
            <EngOnly><JsonInspector data={{ dataset_definition: run.data.dataset_definition, bootstrap: m.bootstrap, uncalibrated_summary: m.uncalibrated_summary }} label="run metadata" /></EngOnly>
          </div>
        )}
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Feature ablation" subtitle={ablRun.data?.metrics?.note || "Drop-one-group, scored on validation only"}
          right={ablations.length > 0 && <select className="input" value={abl ?? ablations[0].evaluation_run_id} onChange={(e) => setAbl(Number(e.target.value))}>
            {ablations.map((a) => <option key={a.evaluation_run_id} value={a.evaluation_run_id}>#{a.evaluation_run_id} {a.family}{a.model_status === "archived" ? " (archived model)" : ""}</option>)}</select>}>
          {ablations.length === 0 && <Empty>No ablation runs yet.</Empty>}
          {ablRun.data && (() => {
            const rows = ablRun.data.metrics.results.filter((r: any) => r.config !== "all_features");
            const ref = ablRun.data.metrics.results.find((r: any) => r.config === "all_features");
            return (
              <>
                <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 24)}>
                  <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, left: 40, bottom: 0 }}>
                    {grid}<XAxis type="number" {...axis} /><YAxis type="category" dataKey="config" {...axis} interval={0} width={150} tickFormatter={(v) => v.replace("without_", "− ")} />
                    <ReferenceLine x={0} stroke="#64748b" />
                    <Tooltip {...tooltipStyle} formatter={(v: number) => v.toFixed(4)} />
                    <Bar dataKey="delta_log_loss" isAnimationActive={false}>{rows.map((r: any) => <Cell key={r.config} fill={r.delta_log_loss > 0 ? "#22d3ee" : "#f87171"} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="text-[11px] text-muted tnum">Δ validation log loss when the group is removed (positive = group helps). Reference: {ref ? `${num(ref.val_log_loss, 4)} LL, ${pct(ref.val_top1)} top-1` : "—"}; {ablRun.data.metrics.seeds} seeds.</div>
              </>
            );
          })()}
        </Card>
        <Card title="Walk-forward evaluation" subtitle="Expanding-window refits through the test season (hyperparameters frozen)">
          {walk.length === 0 ? <Empty>No walk-forward runs yet. Queue a <code>walk_forward</code> job in the Engineering Console.</Empty> : (
            <Table><thead><tr><th className="th">Run</th><th className="th">Model</th><th className="th">n</th><th className="th">Top-1</th><th className="th">Log loss</th><th className="th">Brier</th><th className="th" /></tr></thead>
              <tbody>{walk.map((w) => (<tr key={w.evaluation_run_id}><td className="td">#{w.evaluation_run_id}</td><td className="td">{w.family}</td><td className="td">{w.n_samples}</td>
                <td className="td">{pct(w.summary?.top1_accuracy)}</td><td className="td">{num(w.summary?.log_loss)}</td><td className="td">{num(w.summary?.brier)}</td>
                <td className="td"><button className="btn-ghost text-xs" onClick={() => setRunId(w.evaluation_run_id)}>Detail</button></td></tr>))}</tbody></Table>
          )}
          <div className="mt-3 text-[11px] text-muted">Pitch-type colors: {["FF", "SL", "CH", "SI"].map((c) => <span key={c} style={{ color: pitchColor(c) }} className="mr-2">{c}</span>)}</div>
        </Card>
      </div>
    </div>
  );
}
