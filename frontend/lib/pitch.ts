export const PITCH_COLORS: Record<string, string> = {
  FF: "#f87171",
  SL: "#60a5fa",
  CH: "#4ade80",
  SI: "#fbbf24",
  OTHER: "#c084fc",
};
const FALLBACK = ["#f472b6", "#a3e635", "#fb923c", "#94a3b8"];
export function pitchColor(code: string, i = 0): string {
  return PITCH_COLORS[code] || FALLBACK[i % FALLBACK.length];
}
export const PITCH_NAMES: Record<string, string> = {
  FF: "Four-Seam Fastball",
  SI: "Sinker",
  FC: "Cutter",
  SL: "Slider",
  ST: "Sweeper",
  CU: "Curveball",
  CH: "Changeup",
  FS: "Split-Finger",
  OTHER: "Other (grouped)",
};
export const pitchName = (c?: string | null) => (c ? PITCH_NAMES[c] || c : "—");

export const pct = (x?: number | null, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(d)}%`);
export const num = (x?: number | null, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(d));
export const ci = (c?: number[] | null, d = 1) => (c && c.length === 2 ? `${pct(c[0], d)} – ${pct(c[1], d)}` : "—");
