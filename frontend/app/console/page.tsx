"use client";
import { useEffect, useState } from "react";
import { api, ApiError, useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { Badge, Card, Empty, ErrorBox, Field, JsonInspector, Loading, Note, PageHeader, Table } from "@/components/ui";
import { TraceTimeline } from "@/components/trace";
import { Job, JobProgress } from "@/components/jobs";

export default function ConsolePage() {
  const { pid, engineering } = useApp();
  const traces = useApi<any[]>("/traces?limit=40");
  const [tid, setTid] = useState<string | null>(null);
  const trace = useApi<any>(tid ? `/traces/${tid}` : null);
  useEffect(() => { if (!tid && traces.data?.length) setTid(traces.data[0].trace_id); }, [traces.data, tid]);
  const jobs = useApi<Job[]>("/jobs?limit=40");
  const kinds = useApi<Record<string, string>>("/jobs/kinds");
  const validation = useApi<any>(pid ? `/pitchers/${pid}/validation` : null);
  const [kind, setKind] = useState("benchmark");
  const [params, setParams] = useState("{}");
  const [err, setErr] = useState<ApiError | Error | null>(null);
  const [openJob, setOpenJob] = useState<number | null>(null);
  const jobDetail = useApi<Job>(openJob ? `/jobs/${openJob}` : null);
  useEffect(() => {
    const active = jobs.data?.some((j) => j.status === "running" || j.status === "queued");
    if (!active) return;
    const t = setInterval(() => jobs.reload(), 2000);
    return () => clearInterval(t);
  }, [jobs.data, jobs]);
  async function enqueue() {
    setErr(null);
    try { await api("/jobs", { json: { kind, params: JSON.parse(params || "{}") } }); jobs.reload(); } catch (e) { setErr(e as Error); }
  }
  async function act(id: number, a: "cancel" | "retry") {
    try { await api(`/jobs/${id}/${a}`, { method: "POST" }); jobs.reload(); } catch (e) { setErr(e as ApiError); }
  }
  return (
    <div className="space-y-5">
      <PageHeader title="Engineering Observability Console" subtitle="Recorded processing stages of real predictions, background jobs, and data validation" />
      {!engineering && <Note>Tip: switch to Engineering view (top right) to expose trace JSON and model internals across all pages.</Note>}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
        <Card title="Recent pipeline traces" right={<button className="btn-ghost text-xs" onClick={() => traces.reload()}>Refresh</button>}>
          <ErrorBox error={traces.error} />
          {traces.data?.length === 0 && <Empty>No traces yet — make a prediction.</Empty>}
          <div className="space-y-1 max-h-[34rem] overflow-auto">
            {traces.data?.map((t) => (
              <button key={t.trace_id} onClick={() => setTid(t.trace_id)} className={`w-full text-left rounded-md border px-2 py-1.5 text-xs ${t.trace_id === tid ? "border-accent bg-accent/5" : "border-edge hover:bg-white/5"}`}>
                <div className="flex items-center gap-2"><Badge tone={t.status}>{t.status}</Badge><span>{t.kind}</span><span className="ml-auto tnum text-muted">{t.total_ms.toFixed(0)} ms</span></div>
                <div className="text-muted mt-0.5 tnum">{new Date(t.created_at).toLocaleString()} · {t.n_stages} stages</div>
              </button>
            ))}
          </div>
        </Card>
        <Card title="Trace detail" subtitle="Each stage is recorded by the backend while it runs; nothing here is simulated.">
          <ErrorBox error={trace.error} />
          {trace.loading && !trace.data && <Loading />}
          {trace.data && <><TraceTimeline trace={trace.data} /><div className="mt-3"><JsonInspector data={trace.data} label="raw trace JSON" /></div></>}
        </Card>
      </div>
      <Card title="Background jobs" subtitle="Postgres-backed queue; run `python -m pitchiq.jobs.worker` to process.">
        <div className="flex flex-wrap gap-2 items-end mb-3">
          <Field label="Job kind"><select className="input" value={kind} onChange={(e) => setKind(e.target.value)}>
            {Object.entries(kinds.data || {}).map(([k, d]) => <option key={k} value={k}>{k} — {d}</option>)}</select></Field>
          <Field label="Params (JSON)"><input className="input font-mono w-64" value={params} onChange={(e) => setParams(e.target.value)} /></Field>
          <button className="btn-primary" onClick={enqueue}>Enqueue</button>
          <button className="btn-ghost" onClick={() => jobs.reload()}>Refresh</button>
        </div>
        <ErrorBox error={err || jobs.error} />
        {jobs.data?.length === 0 && <Empty>No jobs yet.</Empty>}
        {jobs.data && jobs.data.length > 0 && (
          <Table>
            <thead><tr><th className="th">#</th><th className="th">Kind</th><th className="th">Status</th><th className="th">Progress</th><th className="th">Records</th><th className="th">Message / error</th><th className="th">Created</th><th className="th">Finished</th><th className="th" /></tr></thead>
            <tbody>{jobs.data.map((j) => (
              <tr key={j.job_id}>
                <td className="td">{j.job_id}</td><td className="td">{j.kind}</td><td className="td"><Badge tone={j.status}>{j.status}</Badge></td>
                <td className="td w-32"><div className="h-1.5 rounded bg-white/5"><div className="h-full rounded bg-accent" style={{ width: `${j.progress * 100}%` }} /></div></td>
                <td className="td">{j.records_processed}</td>
                <td className="td max-w-[18rem] truncate" title={j.error || j.message || ""}>{j.error ? <span className="text-red-300">{j.error.split("\n")[0]}</span> : j.message}</td>
                <td className="td text-muted">{new Date(j.created_at).toLocaleString()}</td><td className="td text-muted">{j.finished_at ? new Date(j.finished_at).toLocaleTimeString() : "—"}</td>
                <td className="td"><div className="flex gap-1">
                  <button className="btn-ghost text-xs" onClick={() => setOpenJob(openJob === j.job_id ? null : j.job_id)}>Detail</button>
                  {(j.status === "queued" || j.status === "running") && <button className="btn-danger text-xs" onClick={() => act(j.job_id, "cancel")}>Cancel</button>}
                  {(j.status === "failed" || j.status === "cancelled") && <button className="btn-ghost text-xs" onClick={() => act(j.job_id, "retry")}>Retry</button>}
                </div></td>
              </tr>))}</tbody>
          </Table>
        )}
        {openJob && jobDetail.data && (
          <div className="mt-3 space-y-2"><JobProgress job={jobDetail.data} /><JsonInspector data={jobDetail.data} label={`job #${openJob}`} defaultOpen /></div>
        )}
      </Card>
      <Card title="Data validation report" subtitle="Latest automated ingestion checks">
        <ErrorBox error={validation.error} />
        {validation.data === null && !validation.loading && <Empty>No report.</Empty>}
        {validation.data && (
          <>
            <div className="flex flex-wrap gap-3 text-sm tnum mb-3">
              <span>pitches <b>{validation.data.summary.pitches}</b></span><span>games <b>{validation.data.summary.games}</b></span>
              <span>issues <b>{validation.data.summary.issues}</b></span><span>missing pitch type <b>{validation.data.summary.missing_pitch_type}</b></span>
              {Object.entries(validation.data.summary.by_type || {}).map(([k, v]) => <Badge key={k} tone="warning">{k}: {String(v)}</Badge>)}
              <span className="text-muted">{new Date(validation.data.created_at).toLocaleString()}</span>
            </div>
            {validation.data.issues.length > 0 && (
              <Table><thead><tr><th className="th">Type</th><th className="th">Game</th><th className="th">Detail</th></tr></thead>
                <tbody>{validation.data.issues.slice(0, 50).map((i: any, k: number) => (
                  <tr key={k}><td className="td">{i.type}</td><td className="td">{i.game_pk}</td><td className="td text-xs text-muted font-mono">{JSON.stringify(Object.fromEntries(Object.entries(i).filter(([x]) => !["type", "game_pk"].includes(x))))}</td></tr>))}</tbody></Table>
            )}
          </>
        )}
      </Card>
    </div>
  );
}
