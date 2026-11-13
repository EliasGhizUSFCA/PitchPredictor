"use client";
import { useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { Badge, Card, ErrorBox, Field, PageHeader } from "@/components/ui";

export default function SettingsPage() {
  const { pitchers, pid, setPid, engineering, setEngineering } = useApp();
  const health = useApi<any>("/health");
  return (
    <div className="space-y-5 max-w-3xl">
      <PageHeader title="Settings" />
      <Card title="API status">
        <ErrorBox error={health.error} what="Backend" />
        {health.data && <div className="flex gap-3 text-sm items-center"><Badge tone={health.data.status === "ok" ? "ok" : "warning"}>{health.data.status}</Badge><span>database: {health.data.database}</span></div>}
        <p className="text-xs text-muted mt-2">The frontend proxies <code>/api/*</code> to <code>PITCHIQ_API_URL</code> (default http://localhost:8000).</p>
      </Card>
      <Card title="Preferences">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Pitcher"><select className="input" value={pid ?? ""} onChange={(e) => setPid(Number(e.target.value))}>
            {pitchers.map((p) => <option key={p.pitcher_id} value={p.pitcher_id}>{p.full_name} ({p.pitches} pitches)</option>)}</select></Field>
          <Field label="View mode"><select className="input" value={engineering ? "e" : "s"} onChange={(e) => setEngineering(e.target.value === "e")}>
            <option value="s">Standard</option><option value="e">Engineering</option></select></Field>
        </div>
      </Card>
      <Card title="Data sources & video rights">
        <ul className="text-sm text-slate-300 space-y-2 list-disc pl-5">
          <li>Pitch-by-pitch data: Baseball Savant Statcast search (regular season). Pitch classifications are the official Statcast labels.</li>
          <li>Lineups, batter metadata and per-pitch event ids (playId): MLB Stats API live game feeds.</li>
          <li>Video: PitchIQ links to the official Baseball Savant per-pitch video page (external playback). It does not download, re-host, or embed MLB footage. In-app playback is only enabled for operator-licensed clips registered in a manifest.</li>
          <li>All probabilities shown are outputs of trained models; all accuracy figures come from stored evaluation records. Simulator results are sensitivity analyses, not causal estimates.</li>
        </ul>
      </Card>
    </div>
  );
}
