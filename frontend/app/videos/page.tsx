"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError, qs, useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { Badge, Card, Empty, ErrorBox, Field, Loading, PageHeader, Stat, Table } from "@/components/ui";
import { PitchChip } from "@/components/pitch";
import { JobButton } from "@/components/jobs";

const LEGEND = [
  ["official_embed", "Available in an authorized embedded player (Mode A) — none configured"],
  ["external_link", "Official external link (Mode B) — opens Baseball Savant after reveal"],
  ["licensed_local", "Licensed local clip (Mode A) — operator-supplied with licence reference"],
  ["metadata_only", "Metadata only — reference found, no playable video confirmed"],
  ["no_match", "No verified match — challenge falls back to data-only (Mode C)"],
];

type Filters = Record<string, string | undefined>;

export default function VideosPage() {
  const { pid, pitcher } = useApp();
  const router = useRouter();
  const [f, setF] = useState<Filters>({});
  const [page, setPage] = useState(0);
  const [showLabels, setShowLabels] = useState(false);
  const limit = 25;
  const cov = useApi<any>(pid ? `/pitchers/${pid}/videos/coverage` : null);
  const opts = useApi<any>(pid ? `/pitchers/${pid}/filters` : null);
  const rows = useApi<any>(pid ? `/pitchers/${pid}/videos${qs({ ...f, limit, offset: page * limit })}` : null, [JSON.stringify(f), page]);
  const review = useApi<any>(pid ? `/pitchers/${pid}/videos?match_confidence=strong_review&limit=50` : null);
  const [err, setErr] = useState<ApiError | null>(null);
  const set = (k: string) => (v: string) => { setPage(0); setF({ ...f, [k]: v || undefined }); };
  const c = cov.data;

  async function randomClip() {
    try { const r = await api<any>(`/pitchers/${pid}/videos/random`); router.push(`/?pitch=${r.pitch_uid}`); } catch (e) { setErr(e as ApiError); }
  }
  async function decide(id: number, decision: "approved" | "rejected") {
    if (!window.confirm(`Mark this match as ${decision}?`)) return;
    try { await api(`/videos/${id}/review`, { json: { decision } }); review.reload(); cov.reload(); rows.reload(); } catch (e) { setErr(e as ApiError); }
  }
  const reload = () => { cov.reload(); rows.reload(); review.reload(); };
  return (
    <div className="space-y-5">
      <PageHeader title={`${pitcher?.full_name ?? ""} Video Library`} subtitle="Automatically discovered official pitch videos, matched to Statcast pitches by MLB playId + multi-field verification"
        right={<div className="flex flex-wrap gap-2 items-start">
          <button className="btn-primary" onClick={randomClip} disabled={!pid}>Random eligible clip → blind predict</button>
          <JobButton kind="build_video_library" label="Find more videos" onDone={reload} />
          <JobButton kind="verify_video_availability" params={{ limit: 50 }} label="Verify availability (50)" onDone={reload} />
        </div>} />
      <ErrorBox error={err || cov.error} />
      {c && (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
          <Stat label="Historical pitches" value={c.total_pitches.toLocaleString()} />
          <Stat label="With video reference" value={c.with_video_reference.toLocaleString()} />
          <Stat label="Verified pitch matches" value={c.verified_matches.toLocaleString()} accent />
          <Stat label="In-app playback" value={c.in_app_playback.toLocaleString()} sub="licensed / embed only" />
          <Stat label="External viewing" value={c.external_viewing.toLocaleString()} sub="official Savant page" />
          <Stat label="Availability confirmed" value={c.external_confirmed_available.toLocaleString()} sub={`last check ${c.last_verification ? new Date(c.last_verification).toLocaleString() : "never"}`} />
          <Stat label="Availability unchecked" value={c.availability_unchecked.toLocaleString()} />
          <Stat label="Unavailable" value={c.unavailable.toLocaleString()} />
          <Stat label="Awaiting verification" value={c.awaiting_verification.toLocaleString()} />
          <Stat label="Last discovery" value={c.last_discovery ? new Date(c.last_discovery).toLocaleDateString() : "never"} />
        </div>
      )}
      <Card title="Availability legend">
        <div className="grid gap-1.5 md:grid-cols-2 text-xs">
          {LEGEND.map(([k, d]) => <div key={k} className="flex gap-2 items-start"><Badge tone={k}>{k}</Badge><span className="text-muted">{d}</span></div>)}
        </div>
        <p className="text-[11px] text-muted mt-3">PitchIQ never downloads or re-hosts MLB footage. Metadata references are not represented as downloaded video.</p>
      </Card>
      <Card title="Search clips" right={<label className="flex items-center gap-2 text-xs text-muted"><input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} />Show pitch types (spoilers)</label>}>
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-2 mb-3">
          <Field label="Game date"><input type="date" className="input" value={f.game_date || ""} onChange={(e) => set("game_date")(e.target.value)} /></Field>
          <Field label="Opponent"><input className="input" placeholder="e.g. PHI" value={f.opponent || ""} onChange={(e) => set("opponent")(e.target.value.toUpperCase())} /></Field>
          <Field label="Batter"><select className="input" value={f.batter_id || ""} onChange={(e) => set("batter_id")(e.target.value)}>
            <option value="">All</option>{(opts.data?.batters || []).map((b: any) => <option key={b.batter_id} value={b.batter_id}>{b.name}</option>)}</select></Field>
          <Field label="Bats"><select className="input" value={f.stand || ""} onChange={(e) => set("stand")(e.target.value)}><option value="">Both</option><option>L</option><option>R</option></select></Field>
          <Field label="Inning"><input type="number" min={1} max={12} className="input" value={f.inning || ""} onChange={(e) => set("inning")(e.target.value)} /></Field>
          <Field label="Count"><select className="input" value={f.count || ""} onChange={(e) => set("count")(e.target.value)}><option value="">All</option>{(opts.data?.counts || []).map((x: string) => <option key={x}>{x}</option>)}</select></Field>
          <Field label="Pitch type"><select className="input" value={f.pitch_type || ""} onChange={(e) => set("pitch_type")(e.target.value)}><option value="">All</option>{["FF", "SL", "CH", "SI"].map((x) => <option key={x}>{x}</option>)}</select></Field>
          <Field label="Pitch count ≥"><input type="number" className="input" value={f.pc_min || ""} onChange={(e) => set("pc_min")(e.target.value)} /></Field>
          <Field label="Pitch count ≤"><input type="number" className="input" value={f.pc_max || ""} onChange={(e) => set("pc_max")(e.target.value)} /></Field>
          <Field label="Availability"><select className="input" value={f.availability || ""} onChange={(e) => set("availability")(e.target.value)}>
            <option value="">All</option>{["unchecked", "external_available", "embed_available", "licensed_local", "metadata_only", "unavailable"].map((x) => <option key={x}>{x}</option>)}</select></Field>
          <Field label="Match"><select className="input" value={f.match_confidence || ""} onChange={(e) => set("match_confidence")(e.target.value)}>
            <option value="">All</option>{["exact_verified", "strong_review", "possible", "no_match"].map((x) => <option key={x}>{x}</option>)}</select></Field>
        </div>
        <ErrorBox error={rows.error} />
        {rows.loading && !rows.data && <Loading />}
        {rows.data && rows.data.rows.length === 0 && <Empty>No clips match.</Empty>}
        {rows.data && rows.data.rows.length > 0 && (
          <>
            <Table>
              <thead><tr><th className="th">Game</th><th className="th">Batter</th><th className="th">Inn</th><th className="th">Count</th><th className="th">AB.pitch</th><th className="th">Game #</th>
                {showLabels && <th className="th">Pitch</th>}<th className="th">Availability</th><th className="th">Source</th><th className="th">Match</th><th className="th">Playback</th><th className="th">Link</th></tr></thead>
              <tbody>{rows.data.rows.map((r: any) => (
                <tr key={r.video_asset_id}>
                  <td className="td">{r.game_date} <span className="text-muted">vs {r.opponent}</span></td>
                  <td className="td">{r.batter} <span className="text-muted">({r.stand})</span></td><td className="td">{r.inning}</td><td className="td">{r.count}</td>
                  <td className="td">{r.at_bat_number}.{r.pitch_number}</td><td className="td">{r.game_pitch_seq}</td>
                  {showLabels && <td className="td"><PitchChip code={r.pitch_type} /></td>}
                  <td className="td"><Badge tone={r.availability}>{r.availability}</Badge></td><td className="td text-muted">{r.source_provider}</td>
                  <td className="td"><Badge tone={r.match_confidence}>{r.match_confidence}</Badge></td><td className="td"><Badge>{r.playback_method}</Badge></td>
                  <td className="td">{r.source_page_url ? <a className="text-accent hover:underline" href={r.source_page_url} target="_blank" rel="noopener noreferrer">official ↗</a> : "—"}</td>
                </tr>))}</tbody>
            </Table>
            <div className="flex flex-wrap items-center gap-2 mt-3 text-xs text-muted tnum">
              <button className="btn-ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>Prev</button>
              <span>{page * limit + 1}–{Math.min((page + 1) * limit, rows.data.total)} of {rows.data.total.toLocaleString()}</span>
              <button className="btn-ghost" disabled={(page + 1) * limit >= rows.data.total} onClick={() => setPage(page + 1)}>Next</button>
              <span className="ml-auto">Opening a link before predicting spoils that pitch — use the Challenge for blind tests.</span>
            </div>
          </>
        )}
      </Card>
      <Card title="Verification queue" subtitle="Identifiers match but a secondary check (count / pitch type) disagrees. These never enter the challenge until approved.">
        <ErrorBox error={review.error} />
        {review.data?.rows.length === 0 && <Empty>Queue is empty.</Empty>}
        {review.data && review.data.rows.length > 0 && (
          <Table>
            <thead><tr><th className="th">Game</th><th className="th">Batter</th><th className="th">AB.pitch</th><th className="th">Evidence</th><th className="th">Link</th><th className="th">Decision</th></tr></thead>
            <tbody>{review.data.rows.map((r: any) => (
              <tr key={r.video_asset_id}>
                <td className="td">{r.game_date}</td><td className="td">{r.batter}</td><td className="td">{r.at_bat_number}.{r.pitch_number}</td>
                <td className="td text-xs !whitespace-normal min-w-[14rem]">{Object.entries(r.match_evidence || {}).map(([k, v]) => (
                  <span key={k} className={`mr-1.5 ${v === true ? "text-emerald-300" : v === false ? "text-red-300" : "text-muted"}`}>{k}{v === true ? "✓" : v === false ? "✗" : "?"}</span>))}</td>
                <td className="td">{r.source_page_url ? <a className="text-accent hover:underline" href={r.source_page_url} target="_blank" rel="noopener noreferrer">review ↗</a> : "—"}</td>
                <td className="td"><div className="flex gap-1"><button className="btn-ghost text-xs" onClick={() => decide(r.video_asset_id, "approved")}>Approve</button><button className="btn-danger text-xs" onClick={() => decide(r.video_asset_id, "rejected")}>Reject</button></div></td>
              </tr>))}</tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}
