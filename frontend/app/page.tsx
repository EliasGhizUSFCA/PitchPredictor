"use client";
import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, ApiError, useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { ci, num, pct } from "@/lib/pitch";
import { Badge, Card, EngOnly, Empty, ErrorBox, JsonInspector, Loading, PageHeader, Stat, Table } from "@/components/ui";
import { PitchChip, ProbBars, Pred } from "@/components/pitch";
import { AttributionPanel, Attribution } from "@/components/explain";
import { SituationPanel, Situation } from "@/components/situation";
import { LockedVideo, RevealDetails, RevealPayload, VideoReveal } from "@/components/reveal";
import { Trace, TraceTimeline } from "@/components/trace";

type PredictResp = {
  model_version: string; model_family: string; calibration_status: string; predictions: Pred[]; attribution: Attribution;
  inference_ms: number; prediction_id: number; pitch_uid: number; input_features: Record<string, unknown>; trace: Trace; situation: Situation;
};

export default function Page() {
  return <Suspense fallback={<Loading />}><Dashboard /></Suspense>;
}

function Dashboard() {
  const { pid } = useApp();
  const params = useSearchParams();
  const summary = useApi<any>(pid ? `/pitchers/${pid}/summary` : null);
  const games = useApi<any[]>(pid ? `/pitchers/${pid}/games?split=test` : null);
  const [gamePk, setGamePk] = useState<number | null>(null);
  const pitches = useApi<any[]>(pid && gamePk ? `/pitchers/${pid}/games/${gamePk}/pitches` : null);
  const [uid, setUid] = useState<number | null>(null);
  const [pred, setPred] = useState<PredictResp | null>(null);
  const [reveal, setReveal] = useState<RevealPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);
  const history = useApi<any[]>(pid ? `/pitchers/${pid}/predictions?limit=15` : null);

  // Deep link from Video Library "random eligible clip": /?pitch=<uid>
  useEffect(() => {
    const p = Number(params.get("pitch"));
    if (p) {
      api<Situation>(`/pitches/${p}/situation`).then((s) => { setGamePk(s.game_pk); setUid(p); }).catch((e) => setErr(e));
    }
  }, [params]);
  useEffect(() => {
    if (!gamePk && games.data?.length && !params.get("pitch")) setGamePk(games.data[0].game_pk);
  }, [games.data, gamePk, params]);
  useEffect(() => {
    if (pitches.data?.length && (uid === null || !pitches.data.some((p) => p.pitch_uid === uid))) {
      const first = pitches.data.find((p) => !p.automatic_call);
      setUid(first ? first.pitch_uid : null);
    }
  }, [pitches.data, uid]);
  useEffect(() => { setPred(null); setReveal(null); setErr(null); }, [uid]);

  const list = pitches.data || [];
  const idx = list.findIndex((p) => p.pitch_uid === uid);
  const cur = idx >= 0 ? list[idx] : null;

  async function predict() {
    if (!pid || !uid) return;
    setBusy(true); setErr(null); setReveal(null);
    try {
      setPred(await api<PredictResp>(`/pitchers/${pid}/pitches/${uid}/predict`, { method: "POST" }));
      history.reload();
    } catch (e) { setErr(e as ApiError); } finally { setBusy(false); }
  }
  async function doReveal() {
    if (!pred) return;
    setBusy(true);
    try { setReveal(await api<RevealPayload>(`/predictions/${pred.prediction_id}/reveal`, { method: "POST" })); history.reload(); }
    catch (e) { setErr(e as ApiError); } finally { setBusy(false); }
  }
  function step(d: number) {
    let j = idx + d;
    while (list[j] && list[j].automatic_call) j += d;
    if (list[j]) setUid(list[j].pitch_uid);
  }

  const s = summary.data;
  const top = pred?.predictions[0];
  return (
    <div className="space-y-5">
      <PageHeader title={s ? `${s.pitcher.full_name} — Next-Pitch Prediction` : "Dashboard"}
        subtitle={s ? `${s.pitches.toLocaleString()} Statcast pitches · ${s.games} games · ${s.date_range[0]} → ${s.date_range[1]} · classes ${s.classes?.join(", ")}` : undefined} />
      <ErrorBox error={summary.error} what="Summary" />
      {s && (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
          <Stat label="Train / Val / Test" value={`${s.splits.train ?? 0} / ${s.splits.validation ?? 0} / ${s.splits.test ?? 0}`} sub="labeled pitches, chronological by season" />
          <Stat label="Promoted model" value={s.promoted_model?.family ?? "none"} sub={s.promoted_model ? "selected on validation log loss" : "train models first"} />
          <Stat label="Official test top-1" accent value={pct(s.official_benchmark?.top1_accuracy)}
            sub={s.official_benchmark ? `game-bootstrap 95% CI ${ci(s.official_benchmark.bootstrap?.top1_accuracy_ci95)} · n=${s.official_benchmark.n}` : "no benchmark yet"} />
          <Stat label="Test log loss / Brier" value={s.official_benchmark ? `${num(s.official_benchmark.log_loss)} / ${num(s.official_benchmark.brier)}` : "—"}
            sub={s.official_benchmark ? `top-2 ${pct(s.official_benchmark.top2_accuracy)}` : undefined} />
          <Stat label="Verified pitch videos" value={s.video_coverage?.verified_matches?.toLocaleString() ?? "—"}
            sub={`of ${s.video_coverage?.total_pitches?.toLocaleString() ?? "?"} pitches · ${s.video_coverage?.awaiting_verification ?? 0} awaiting review`} />
          <Stat label="Data validation" value={s.validation ? `${s.validation.issues} issues` : "—"}
            sub={s.validation ? `${s.validation.missing_pitch_type} automatic (pitch-clock) calls` : undefined} />
        </div>
      )}

      <Card title="Prediction replay — held-out test games" subtitle="Pitch list is blind; the model predicts from pre-pitch state only, the prediction is saved, then you reveal."
        right={
          <div className="flex flex-wrap gap-2 items-center">
            <select className="input max-w-[16rem]" value={gamePk ?? ""} onChange={(e) => { setGamePk(Number(e.target.value)); setUid(null); }}>
              {(games.data || []).map((g) => <option key={g.game_pk} value={g.game_pk}>{g.game_date} · {g.away_team} @ {g.home_team} ({g.pitches})</option>)}
            </select>
            <select className="input max-w-[18rem]" value={uid ?? ""} onChange={(e) => setUid(Number(e.target.value))}>
              {list.map((p) => <option key={p.pitch_uid} value={p.pitch_uid} disabled={p.automatic_call}>
                #{p.game_pitch_seq} · {p.half === "Top" ? "▲" : "▼"}{p.inning} · {p.batter} ({p.stand}) · {p.count}{p.automatic_call ? " · auto call" : ""}
              </option>)}
            </select>
            <button className="btn-ghost" onClick={() => step(-1)} disabled={idx <= 0}>◀</button>
            <button className="btn-ghost" onClick={() => step(1)} disabled={idx < 0 || idx >= list.length - 1}>▶</button>
            <button className="btn-primary" onClick={predict} disabled={!uid || busy || cur?.automatic_call}>Predict next pitch</button>
          </div>
        }>
        <ErrorBox error={games.error || pitches.error || err} />
        {!pred && !busy && <Empty>{cur ? `Ready: pitch #${cur.game_pitch_seq} to ${cur.batter}. Press “Predict next pitch”.` : "Select a game and pitch."}</Empty>}
        {busy && !pred && <Loading label="Running model inference…" />}
        {pred && (
          <div className="grid gap-4 xl:grid-cols-[minmax(0,18rem)_minmax(0,1fr)_minmax(0,20rem)]">
            <div className="card p-3 bg-panel2/50 order-2 xl:order-1"><div className="label mb-2">Game situation (pre-pitch)</div><SituationPanel s={pred.situation} /></div>
            <div className="order-1 xl:order-2 space-y-3">
              {reveal ? <VideoReveal v={reveal.video_reference} /> : <LockedVideo />}
              {reveal ? <RevealDetails r={reveal} /> : (
                <div className="flex flex-wrap items-center gap-3">
                  <button className="btn-primary" onClick={doReveal} disabled={busy}>Reveal pitch</button>
                  <span className="text-xs text-muted">Prediction #{pred.prediction_id} is saved; the actual pitch has not been loaded.</span>
                </div>
              )}
            </div>
            <div className="card p-3 bg-panel2/50 order-3 space-y-3">
              <div className="label">Next-pitch probabilities (model output)</div>
              {top && <div className="flex items-center gap-2"><PitchChip code={top.pitch} size="lg" /><div><div className="font-semibold">{top.pitch_name}</div><div className="text-xs text-muted tnum">{pct(top.probability)} highest probability</div></div></div>}
              <ProbBars predictions={pred.predictions} actual={reveal?.actual.model_class} />
              <div className="text-[11px] text-muted space-y-0.5">
                <div>Model <span className="font-mono text-slate-300 break-all">{pred.model_version}</span></div>
                <div>Calibration: {pred.calibration_status}</div>
                <div className="tnum">Inference {pred.inference_ms.toFixed(1)} ms · end-to-end {pred.trace.total_ms.toFixed(1)} ms</div>
              </div>
            </div>
          </div>
        )}
      </Card>

      {pred && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Feature contributions" subtitle="Real attribution from the trained model for this prediction">
            <AttributionPanel attribution={pred.attribution} input={pred.input_features} />
          </Card>
          <Card title="Model inputs & pipeline" subtitle="Pre-pitch feature vector actually sent to the model">
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 text-xs mb-3">
              {["stand", "bat_order_slot", "count_str", "inning", "pitch_count_before", "times_through_order", "outs", "score_diff", "prev1_type", "prev2_type", "prev1_result", "bvp_n"].map((k) => (
                <div key={k} className="bg-panel2 rounded px-2 py-1 border border-edge"><div className="text-muted text-[10px]">{k}</div><div className="tnum">{String(pred.input_features[k] ?? "null")}</div></div>
              ))}
            </div>
            <EngOnly>
              <div className="space-y-2">
                <TraceTimeline trace={pred.trace} />
                <JsonInspector data={pred.input_features} label="full input_features" />
              </div>
            </EngOnly>
            <div className="text-[11px] text-muted">Switch to Engineering view for the full pipeline trace and JSON.</div>
          </Card>
        </div>
      )}

      <Card title="Prediction history" subtitle="Interactive predictions (dashboard, challenge). Benchmark predictions are in Model Performance.">
        <ErrorBox error={history.error} />
        {history.data && history.data.length === 0 && <Empty>No predictions yet.</Empty>}
        {history.data && history.data.length > 0 && (
          <Table>
            <thead><tr><th className="th">#</th><th className="th">Context</th><th className="th">Pitch</th><th className="th">Top pick</th><th className="th">P(top)</th><th className="th">Actual</th><th className="th">P(actual)</th><th className="th">Result</th><th className="th">ms</th></tr></thead>
            <tbody>
              {history.data.map((h) => (
                <tr key={h.prediction_id}>
                  <td className="td text-muted">{h.prediction_id}</td><td className="td"><Badge>{h.context}</Badge></td><td className="td">{h.pitch_uid}</td>
                  <td className="td"><PitchChip code={h.top_pitch} /></td><td className="td">{pct(h.probabilities[h.top_pitch])}</td>
                  <td className="td">{h.actual_pitch ? <PitchChip code={h.actual_pitch} /> : <span className="text-muted">not revealed</span>}</td>
                  <td className="td">{h.prob_actual != null ? pct(h.prob_actual) : "—"}</td>
                  <td className="td">{h.is_correct_top1 == null ? "—" : h.is_correct_top1 ? <Badge tone="ok">hit</Badge> : <Badge tone="error">miss</Badge>}</td>
                  <td className="td text-muted">{h.inference_ms ? h.inference_ms.toFixed(1) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}
