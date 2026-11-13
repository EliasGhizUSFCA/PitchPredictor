"use client";
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { Badge, ErrorBox } from "./ui";

export type Job = { job_id: number; kind: string; description?: string; status: string; progress: number; records_processed: number;
  message: string | null; error: string | null; created_at: string; started_at: string | null; finished_at: string | null; attempts: number;
  cancel_requested: boolean; params: Record<string, unknown>; result?: unknown; log?: { ts: string; msg: string }[] };

export function useJobPoll(jobId: number | null, onDone?: () => void) {
  const [job, setJob] = useState<Job | null>(null);
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    let t: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const j = await api<Job>(`/jobs/${jobId}`);
        if (!alive) return;
        setJob(j);
        if (["queued", "running"].includes(j.status)) t = setTimeout(tick, 1500);
        else onDone?.();
      } catch { if (alive) t = setTimeout(tick, 3000); }
    };
    tick();
    return () => { alive = false; clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);
  return job;
}

export function JobButton({ kind, params = {}, label, onDone }: { kind: string; params?: Record<string, unknown>; label: string; onDone?: () => void }) {
  const [jobId, setJobId] = useState<number | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const job = useJobPoll(jobId, onDone);
  const running = job && ["queued", "running"].includes(job.status);
  return (
    <div className="space-y-1.5 min-w-[12rem]">
      <button className="btn-ghost w-full justify-center" disabled={!!running}
        onClick={async () => { setErr(null); try { setJobId((await api<Job>("/jobs", { json: { kind, params } })).job_id); } catch (e) { setErr(e as ApiError); } }}>
        {label}
      </button>
      <ErrorBox error={err} />
      {job && <JobProgress job={job} />}
    </div>
  );
}

export function JobProgress({ job }: { job: Job }) {
  return (
    <div className="text-[11px] space-y-1">
      <div className="flex items-center gap-2"><Badge tone={job.status}>{job.status}</Badge><span className="text-muted">job #{job.job_id}</span><span className="ml-auto tnum">{Math.round(job.progress * 100)}%</span></div>
      <div className="h-1.5 rounded bg-white/5"><div className="h-full rounded bg-accent transition-all" style={{ width: `${job.progress * 100}%` }} /></div>
      {job.message && <div className="text-muted truncate" title={job.message}>{job.message}</div>}
      {job.status === "queued" && <div className="text-amber-300/80">Waiting for a worker (python -m pitchiq.jobs.worker).</div>}
      {job.error && <div className="text-red-300 truncate" title={job.error}>{job.error.split("\n")[0]}</div>}
    </div>
  );
}
