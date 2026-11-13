"use client";
import { useEffect, useRef } from "react";
import { num, pct, pitchName } from "@/lib/pitch";
import { PitchChip } from "./pitch";
import { Badge } from "./ui";

export type RevealPayload = {
  actual: { pitch_type: string; pitch_name: string; model_class: string; release_speed: number | null; spin_rate: number | null;
    result: string | null; event: string | null; description: string | null; plate_x: number | null; plate_z: number | null; source: string };
  score: { top1_correct: boolean | null; prob_actual: number | null; rank_actual: number | null; user_guess_correct: boolean | null };
  video_reference: null | { video_asset_id: number; source_provider: string; playback_method: string; source_page_url: string | null;
    authorized_embed_reference: string | null; media_url: string | null; pre_pitch_pause_time: number | null;
    release_frame_timestamp: number | null; match_confidence: string; rights: string; availability: string };
};

export function LockedVideo({ mode, available }: { mode?: string; available?: boolean }) {
  return (
    <div className="aspect-video w-full rounded-lg border border-edge bg-gradient-to-br from-slate-900 to-black flex flex-col items-center justify-center text-center p-4">
      <div className="text-3xl mb-2">🔒</div>
      <div className="text-sm font-medium">Clip locked until the prediction is frozen and revealed</div>
      <div className="text-xs text-muted mt-1">
        {available === false || mode === "data_only" ? "No verified clip for this pitch — data-only challenge." :
          mode === "external" ? "Verified official clip available via Baseball Savant (opens after reveal)." :
          mode === "in_app" ? "Licensed clip will play in-app, paused before delivery." : "Video reference is withheld before reveal."}
      </div>
    </div>
  );
}

export function VideoReveal({ v }: { v: RevealPayload["video_reference"] }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (ref.current && v?.pre_pitch_pause_time != null) ref.current.currentTime = v.pre_pitch_pause_time;
  }, [v]);
  if (!v) {
    return (
      <div className="aspect-video w-full rounded-lg border border-edge bg-black/40 flex items-center justify-center text-sm text-muted">
        Data-only (no verified clip)
      </div>
    );
  }
  if (v.playback_method === "licensed_local" && v.media_url) {
    return (
      <div>
        <video ref={ref} src={v.media_url} controls className="w-full rounded-lg border border-edge bg-black" />
        <div className="text-[11px] text-muted mt-1">Licensed local clip · {v.rights}</div>
      </div>
    );
  }
  return (
    <div className="aspect-video w-full rounded-lg border border-edge bg-gradient-to-br from-slate-900 to-black flex flex-col items-center justify-center gap-3 p-4 text-center">
      <div className="text-sm">Official clip hosted by MLB / Baseball Savant</div>
      {v.source_page_url && (
        <a href={v.source_page_url} target="_blank" rel="noopener noreferrer" className="btn-primary">▶ Open official clip (Baseball Savant)</a>
      )}
      <div className="flex flex-wrap justify-center gap-1.5">
        <Badge tone={v.match_confidence}>{v.match_confidence}</Badge>
        <Badge>{v.playback_method}</Badge>
        <Badge>availability: {v.availability}</Badge>
      </div>
      <div className="text-[11px] text-muted max-w-sm">Link-only: PitchIQ does not download, re-host or embed this footage ({v.rights}).</div>
    </div>
  );
}

export function RevealDetails({ r }: { r: RevealPayload }) {
  const a = r.actual;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <PitchChip code={a.pitch_type} size="lg" />
        <div>
          <div className="font-semibold">{a.pitch_name || pitchName(a.pitch_type)}</div>
          <div className="text-[11px] text-muted">Official classification · {a.source}{a.model_class !== a.pitch_type ? ` · model class ${a.model_class}` : ""}</div>
        </div>
        <div className="ml-auto">
          {r.score.top1_correct ? <Badge tone="ok">Top-1 correct</Badge> : <Badge tone="error">Top-1 missed</Badge>}
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-sm">
        <Mini label="P(actual)" v={pct(r.score.prob_actual)} />
        <Mini label="Rank of actual" v={r.score.rank_actual ?? "—"} />
        <Mini label="Velocity" v={a.release_speed ? `${a.release_speed.toFixed(1)} mph` : "—"} />
        <Mini label="Spin" v={a.spin_rate ? `${Math.round(a.spin_rate)} rpm` : "—"} />
        <Mini label="Result" v={(a.result || "—").replaceAll("_", " ")} />
        <Mini label="Event" v={(a.event || "—").replaceAll("_", " ")} />
        <Mini label="Plate x / z" v={`${num(a.plate_x, 2)} / ${num(a.plate_z, 2)}`} />
        {r.score.user_guess_correct !== null && <Mini label="Your guess" v={r.score.user_guess_correct ? "correct" : "wrong"} />}
      </div>
      {a.description && <div className="text-xs text-muted">{a.description}</div>}
    </div>
  );
}

function Mini({ label, v }: { label: string; v: React.ReactNode }) {
  return <div className="bg-panel2 rounded-md px-2 py-1.5 border border-edge"><div className="label">{label}</div><div className="tnum truncate">{v}</div></div>;
}
