"use client";
import { PitchChip } from "./pitch";

export type Situation = {
  pitch_uid: number; game_pk: number; game_date: string; season: number; pitcher_team: string; opponent: string; is_home: boolean;
  batter: { id: number; name: string | null; stand: string }; batting_order_slot: number | null; times_through_order: number | null;
  inning: number; half: string; outs: number; count: { balls: number; strikes: number }; pitch_count_before: number | null;
  pa_pitch_number: number; at_bat_number: number; runners: { on_1b: boolean; on_2b: boolean; on_3b: boolean };
  score: { pitcher_team: number | null; batting_team: number | null };
  previous_pitches: { pitch_type: string; pitch_name: string; result: string; speed: number | null; same_pa: boolean; count: string }[];
};

function Diamond({ r }: { r: Situation["runners"] }) {
  const b = (on: boolean) => (on ? "#22d3ee" : "transparent");
  return (
    <svg viewBox="0 0 60 44" className="w-16 h-12" aria-label="base runners">
      <g stroke="#475569" strokeWidth="1.5">
        <rect x="24" y="2" width="12" height="12" transform="rotate(45 30 8)" fill={b(r.on_2b)} />
        <rect x="42" y="18" width="12" height="12" transform="rotate(45 48 24)" fill={b(r.on_1b)} />
        <rect x="6" y="18" width="12" height="12" transform="rotate(45 12 24)" fill={b(r.on_3b)} />
      </g>
    </svg>
  );
}

export function SituationPanel({ s }: { s: Situation }) {
  const dots = (n: number, total: number, color: string) => (
    <span className="inline-flex gap-1">{Array.from({ length: total }).map((_, i) => (
      <span key={i} className="h-2.5 w-2.5 rounded-full border" style={{ background: i < n ? color : "transparent", borderColor: color }} />
    ))}</span>
  );
  return (
    <div className="space-y-3 text-sm">
      <div>
        <div className="label">Batter</div>
        <div className="font-semibold text-base">{s.batter.name || `#${s.batter.id}`}</div>
        <div className="text-xs text-muted">Bats {s.batter.stand === "L" ? "left" : "right"} · lineup #{s.batting_order_slot ?? "?"} · {ordinal(s.times_through_order)} time through</div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div><div className="label">Inning</div><div className="tnum">{s.half === "Top" ? "▲" : "▼"} {s.inning}</div></div>
        <div><div className="label">Score</div><div className="tnum">{s.pitcher_team} {s.score.pitcher_team ?? "?"} – {s.score.batting_team ?? "?"} {s.opponent}</div></div>
        <div><div className="label">Count</div><div className="text-2xl font-semibold tnum">{s.count.balls}-{s.count.strikes}</div></div>
        <div className="flex items-center gap-2"><Diamond r={s.runners} /><div className="space-y-1 text-xs"><div>B {dots(s.count.balls, 3, "#4ade80")}</div><div>S {dots(s.count.strikes, 2, "#f87171")}</div><div>O {dots(s.outs, 2, "#fbbf24")}</div></div></div>
        <div><div className="label">Pitch count</div><div className="tnum">{s.pitch_count_before ?? "?"} thrown</div></div>
        <div><div className="label">PA pitch #</div><div className="tnum">{s.pa_pitch_number}</div></div>
      </div>
      <div>
        <div className="label mb-1">Previous pitches (most recent first)</div>
        {s.previous_pitches.length === 0 ? <div className="text-xs text-muted">First pitch of the game.</div> : (
          <div className="flex flex-wrap gap-1.5">
            {s.previous_pitches.map((p, i) => (
              <span key={i} className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] border ${p.same_pa ? "border-white/25" : "border-edge opacity-70"}`}
                title={`${p.pitch_name} · ${p.result} · ${p.speed ?? "?"} mph · count ${p.count}${p.same_pa ? " · this PA" : ""}`}>
                <PitchChip code={p.pitch_type} /><span className="text-muted">{p.result.replace("_", " ")}</span>
              </span>
            ))}
          </div>
        )}
        <div className="text-[10px] text-muted mt-1">Outlined = same plate appearance. These pitches were already thrown, so they are pre-pitch information.</div>
      </div>
      <div className="text-[11px] text-muted">{s.game_date} · {s.pitcher_team} {s.is_home ? "vs" : "@"} {s.opponent} · game {s.game_pk}</div>
    </div>
  );
}

function ordinal(n: number | null) {
  if (!n) return "?";
  return n === 1 ? "1st" : n === 2 ? "2nd" : n === 3 ? "3rd" : `${n}th`;
}
