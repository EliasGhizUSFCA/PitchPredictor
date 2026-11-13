"use client";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from "recharts";
import { qs, useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { num, pct, pitchColor } from "@/lib/pitch";
import { Card, ErrorBox, Field, Loading, Note, PageHeader, Table } from "@/components/ui";
import { PitchChip, PitchLegend } from "@/components/pitch";
import { MatrixGrid, UsageStack, axis, tooltipStyle } from "@/components/charts";

export default function AnalyticsPage() {
  const { pid, pitcher } = useApp();
  const [f, setF] = useState<{ season?: string; batter_id?: string; count?: string; game_pk?: string; stand?: string }>({});
  const opts = useApi<any>(pid ? `/pitchers/${pid}/filters` : null);
  const a = useApi<any>(pid ? `/pitchers/${pid}/analytics${qs(f)}` : null, [JSON.stringify(f)]);
  const d = a.data;
  const set = (k: keyof typeof f) => (v: string) => setF({ ...f, [k]: v || undefined });
  const classes: string[] = d?.classes || [];
  const sections: [string, string, string][] = [
    ["by_season", "Pitch usage by season", "season"], ["by_count", "Pitch usage by count", "count"],
    ["by_hand", "Pitch usage by batter handedness", "bats"], ["by_batting_order", "Pitch usage by batting-order position", "slot"],
    ["by_inning", "Pitch usage by inning", "inning"], ["by_pitch_count", "Pitch usage by cumulative pitch count", "pitches"],
    ["by_times_through_order", "Pitch usage by times through the order", "TTO"],
  ];
  return (
    <div className="space-y-5">
      <PageHeader title={`${pitcher?.full_name ?? "Pitcher"} Analytics`} subtitle="Historical pitch selection from official Statcast records" />
      <Note><b>Observed frequencies — not model predictions.</b> {d?.provenance?.source}. Every chart lists its sample size (n).</Note>
      <Card>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <Field label="Season"><select className="input" value={f.season || ""} onChange={(e) => set("season")(e.target.value)}>
            <option value="">All</option>{(opts.data?.seasons || []).map((s: number) => <option key={s} value={s}>{s}</option>)}</select></Field>
          <Field label="Batter hand"><select className="input" value={f.stand || ""} onChange={(e) => set("stand")(e.target.value)}>
            <option value="">Both</option><option value="L">Left</option><option value="R">Right</option></select></Field>
          <Field label="Count"><select className="input" value={f.count || ""} onChange={(e) => set("count")(e.target.value)}>
            <option value="">All</option>{(opts.data?.counts || []).map((c: string) => <option key={c}>{c}</option>)}</select></Field>
          <Field label="Batter"><select className="input" value={f.batter_id || ""} onChange={(e) => set("batter_id")(e.target.value)}>
            <option value="">All</option>{(opts.data?.batters || []).map((b: any) => <option key={b.batter_id} value={b.batter_id}>{b.name} ({b.pitches})</option>)}</select></Field>
          <Field label="Game"><select className="input" value={f.game_pk || ""} onChange={(e) => set("game_pk")(e.target.value)}>
            <option value="">All</option>{(opts.data?.games || []).map((g: any) => <option key={g.game_pk} value={g.game_pk}>{g.game_date} vs {g.opponent}</option>)}</select></Field>
        </div>
        {d && <div className="text-xs text-muted mt-3 tnum">n = {d.n?.toLocaleString()} pitches · {d.games ?? 0} games {d.date_range ? `· ${d.date_range[0]} → ${d.date_range[1]}` : ""}</div>}
      </Card>
      <ErrorBox error={a.error} />
      {a.loading && !d && <Loading />}
      {d && d.n === 0 && <Card><div className="text-sm text-muted">No pitches match these filters.</div></Card>}
      {d && d.n > 0 && (
        <>
          <Card title="Arsenal" right={<PitchLegend classes={classes} />}>
            <Table>
              <thead><tr><th className="th">Pitch</th><th className="th">n</th><th className="th">Share</th><th className="th">Avg velo</th><th className="th">Avg spin</th><th className="th">Whiff / pitch</th></tr></thead>
              <tbody>{d.arsenal.map((r: any) => (
                <tr key={r.pitch_type}><td className="td"><PitchChip code={r.pitch_type} /> <span className="text-muted ml-1">{r.pitch_name}</span></td>
                  <td className="td">{r.n}</td><td className="td">{pct(r.share)}</td><td className="td">{r.velo ? `${r.velo.toFixed(1)} mph` : "—"}</td>
                  <td className="td">{r.spin ? Math.round(r.spin) : "—"}</td><td className="td">{pct(r.whiff_rate)}</td></tr>))}</tbody>
            </Table>
          </Card>
          <div className="grid gap-4 xl:grid-cols-2">
            {sections.map(([k, title, xl]) => (
              <Card key={k} title={title} subtitle="100% stacked share of pitches; hover for exact n">
                <UsageStack rows={d[k].rows} classes={d[k].classes} xLabel={xl} />
              </Card>
            ))}
            <Card title="Pitch-sequence transition matrix" subtitle="Previous pitch → next pitch within the same plate appearance">
              <MatrixGrid rows={d.transitions.from} cols={d.transitions.to} matrix={d.transitions.matrix} rowTitle="previous pitch" colTitle="next pitch" />
              <div className="text-[11px] text-muted mt-2 tnum">First pitch of PA: {Object.entries(d.transitions.first_pitch_of_pa).map(([k, v]) => `${k} ${v}`).join(" · ")}</div>
            </Card>
          </div>
          <Card title="Per-game trends" subtitle="Usage share and average four-seam velocity by game (chronological)">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={d.games_log.map((g: any) => ({ date: g.game_date, ...g.usage, ff_velo: g.ff_velo }))} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                <CartesianGrid stroke="#1e2633" strokeDasharray="3 3" />
                <XAxis dataKey="date" {...axis} minTickGap={30} />
                <YAxis yAxisId="u" {...axis} domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <YAxis yAxisId="v" orientation="right" {...axis} domain={["auto", "auto"]} />
                <Tooltip {...tooltipStyle} formatter={(v: number, n: string) => [n === "ff_velo" ? `${v?.toFixed(1)} mph` : pct(v), n]} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {classes.map((c, i) => <Line key={c} yAxisId="u" dataKey={c} stroke={pitchColor(c, i)} dot={false} strokeWidth={1.5} isAnimationActive={false} />)}
                <Line yAxisId="v" dataKey="ff_velo" name="ff_velo" stroke="#e5e7eb" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
          <div className="grid gap-4 xl:grid-cols-2">
            <Card title="Batter matchups" subtitle="Top 40 batters by pitches seen (filtered)">
              <div className="max-h-[28rem] overflow-auto">
                <Table>
                  <thead><tr><th className="th">Batter</th><th className="th">B</th><th className="th">Pitches</th><th className="th">PA</th><th className="th">K</th><th className="th">BB</th><th className="th">H</th>{classes.map((c) => <th key={c} className="th">{c}</th>)}</tr></thead>
                  <tbody>{d.batters.map((b: any) => (
                    <tr key={`${b.batter_id}-${b.stand}`}><td className="td">{b.name}</td><td className="td">{b.stand}</td><td className="td">{b.pitches}</td><td className="td">{b.plate_appearances}</td>
                      <td className="td">{b.strikeouts}</td><td className="td">{b.walks}</td><td className="td">{b.hits}</td>
                      {classes.map((c) => <td key={c} className="td">{pct(b.usage[c], 0)}</td>)}</tr>))}</tbody>
                </Table>
              </div>
            </Card>
            <Card title="Game log" subtitle="Per-game performance from Statcast events">
              <div className="max-h-[28rem] overflow-auto">
                <Table>
                  <thead><tr><th className="th">Date</th><th className="th">Opp</th><th className="th">Pitches</th><th className="th">BF</th><th className="th">K</th><th className="th">BB</th><th className="th">H</th><th className="th">Whiffs</th><th className="th">FF velo</th></tr></thead>
                  <tbody>{[...d.games_log].reverse().map((g: any) => (
                    <tr key={g.game_pk}><td className="td">{g.game_date}</td><td className="td">{g.opponent}</td><td className="td">{g.pitches}</td><td className="td">{g.batters_faced}</td>
                      <td className="td">{g.strikeouts}</td><td className="td">{g.walks}</td><td className="td">{g.hits}</td><td className="td">{g.whiffs}</td><td className="td">{num(g.ff_velo, 1)}</td></tr>))}</tbody>
                </Table>
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
