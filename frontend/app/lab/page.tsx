"use client";
import { useEffect, useState } from "react";
import { api, ApiError, useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { num, pct } from "@/lib/pitch";
import { Badge, Card, EngOnly, Empty, ErrorBox, Field, JsonInspector, Loading, Note, PageHeader, Table } from "@/components/ui";
import { PitchChip, ProbBars } from "@/components/pitch";
import { AttributionPanel } from "@/components/explain";
import { SituationPanel } from "@/components/situation";

export default function LabPage() {
  const { pid } = useApp();
  const [archived, setArchived] = useState(false);
  const models = useApi<any[]>(pid ? `/pitchers/${pid}/models?include_archived=${archived}` : null, [archived]);
  const games = useApi<any[]>(pid ? `/pitchers/${pid}/games?split=test` : null);
  const [gamePk, setGamePk] = useState<number | null>(null);
  const pitches = useApi<any[]>(pid && gamePk ? `/pitchers/${pid}/games/${gamePk}/pitches` : null);
  const [uid, setUid] = useState<number | null>(null);
  const [mode, setMode] = useState<"pitch" | "state">("pitch");
  const [stateJson, setStateJson] = useState(JSON.stringify({ stand: "R", balls: 0, strikes: 2, inning: 6, pitch_count_before: 80, times_through_order: 3 }, null, 1));
  const [res, setRes] = useState<any[] | null>(null);
  const [sit, setSit] = useState<any>(null);
  const [err, setErr] = useState<ApiError | Error | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (!gamePk && games.data?.length) setGamePk(games.data[0].game_pk); }, [games.data, gamePk]);
  useEffect(() => { if (pitches.data?.length) setUid(pitches.data.find((p) => !p.automatic_call)?.pitch_uid ?? null); }, [pitches.data]);
  async function compare() {
    setBusy(true); setErr(null);
    try {
      let body: any;
      if (mode === "pitch") { body = { pitch_uid: uid }; setSit(await api(`/pitches/${uid}/situation`)); }
      else { body = { state: JSON.parse(stateJson) }; setSit(null); }
      setRes(await api(`/pitchers/${pid}/models/compare`, { json: body }));
    } catch (e) { setErr(e as Error); } finally { setBusy(false); }
  }
  async function promote(v: string) {
    if (!window.confirm(`Promote ${v}?\n\nPromotion should be justified by VALIDATION metrics, not test results.`)) return;
    try { await api(`/models/${v}/promote`, { method: "POST" }); models.reload(); } catch (e) { setErr(e as ApiError); }
  }
  return (
    <div className="space-y-5">
      <PageHeader title="Model Comparison Lab" subtitle="Model registry and side-by-side predictions from every model family for the same situation" />
      <Card title="Model registry" right={<label className="text-xs text-muted flex gap-2 items-center"><input type="checkbox" checked={archived} onChange={(e) => setArchived(e.target.checked)} />show archived</label>}>
        <ErrorBox error={models.error} />
        {models.loading && !models.data && <Loading />}
        {models.data?.length === 0 && <Empty>No models trained.</Empty>}
        {models.data && models.data.length > 0 && (
          <Table>
            <thead><tr><th className="th">Version</th><th className="th">Family</th><th className="th">Status</th><th className="th">Features</th><th className="th">Val top-1</th><th className="th">Val log loss</th><th className="th">Temperature</th><th className="th">Created</th><th className="th" /></tr></thead>
            <tbody>{models.data.map((m) => (
              <tr key={m.model_version}>
                <td className="td font-mono text-[11px]">{m.model_version}</td><td className="td">{m.family}</td><td className="td"><Badge tone={m.status}>{m.status}</Badge></td>
                <td className="td">{m.n_features}</td><td className="td">{pct(m.validation_metrics?.top1_accuracy)}</td><td className="td">{num(m.validation_metrics?.log_loss)}</td>
                <td className="td">{num(m.calibration?.temperature)}</td><td className="td text-muted">{new Date(m.created_at).toLocaleString()}</td>
                <td className="td">{m.status !== "promoted" && m.status !== "archived" && <button className="btn-ghost text-xs" onClick={() => promote(m.model_version)}>Promote</button>}</td>
              </tr>))}</tbody>
          </Table>
        )}
        <EngOnly><div className="mt-3"><JsonInspector data={models.data} label="registry rows" /></div></EngOnly>
      </Card>
      <Card title="Side-by-side predictions">
        <div className="flex flex-wrap gap-3 items-end mb-4">
          <Field label="Input"><select className="input" value={mode} onChange={(e) => setMode(e.target.value as "pitch" | "state")}><option value="pitch">Historical test pitch</option><option value="state">Simulator state (JSON)</option></select></Field>
          {mode === "pitch" ? (
            <>
              <Field label="Game"><select className="input max-w-[16rem]" value={gamePk ?? ""} onChange={(e) => setGamePk(Number(e.target.value))}>
                {(games.data || []).map((g) => <option key={g.game_pk} value={g.game_pk}>{g.game_date} · {g.away_team} @ {g.home_team}</option>)}</select></Field>
              <Field label="Pitch"><select className="input max-w-[18rem]" value={uid ?? ""} onChange={(e) => setUid(Number(e.target.value))}>
                {(pitches.data || []).map((p) => <option key={p.pitch_uid} value={p.pitch_uid} disabled={p.automatic_call}>#{p.game_pitch_seq} · inn {p.inning} · {p.batter} · {p.count}</option>)}</select></Field>
            </>
          ) : (
            <Field label="SimState JSON"><textarea className="input font-mono text-xs w-80 h-24" value={stateJson} onChange={(e) => setStateJson(e.target.value)} /></Field>
          )}
          <button className="btn-primary" onClick={compare} disabled={busy || (mode === "pitch" && !uid)}>Compare all models</button>
        </div>
        <Note>Predictions are shown without the actual pitch. Use the Dashboard or Challenge to reveal outcomes.</Note>
        <div className="mt-3"><ErrorBox error={err} /></div>
        {busy && <Loading label="Running every model…" />}
        {res && (
          <div className="grid gap-4 xl:grid-cols-[minmax(0,16rem)_minmax(0,1fr)] mt-4">
            {sit ? <div className="card p-3 bg-panel2/50"><SituationPanel s={sit} /></div> : <div className="text-xs text-muted">Simulator scenario (no historical situation)</div>}
            <div className="grid gap-3 sm:grid-cols-2">
              {res.map((r) => (
                <div key={r.model_version} className="card p-3 bg-panel2/40">
                  <div className="flex items-center gap-2 mb-2"><span className="font-semibold">{r.family}</span><Badge tone={r.status}>{r.status}</Badge>
                    <span className="ml-auto text-[11px] text-muted tnum">{r.inference_ms.toFixed(2)} ms</span></div>
                  <div className="flex items-center gap-2 mb-2"><PitchChip code={r.predictions[0].pitch} /><span className="tnum text-sm">{pct(r.predictions[0].probability)}</span></div>
                  <ProbBars predictions={r.predictions} compact />
                  <div className="text-[10px] text-muted font-mono mt-2 break-all">{r.model_version}</div>
                  {r.attribution && <details className="mt-2"><summary className="text-xs text-muted cursor-pointer">attribution</summary><div className="mt-2"><AttributionPanel attribution={r.attribution} /></div></details>}
                </div>
              ))}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
