"""Mode A: clips the operator is licensed to store and play locally.

Expects `PITCHIQ_LICENSED_VIDEO_DIR` containing `manifest.csv` with columns:
  file,game_pk,at_bat_number,pitch_number,pitcher_id,batter_id,license_reference
Only files listed in the manifest (with a licence reference) are ever served.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ...config import get_settings
from .base import VideoCandidate, VideoProvider


class LicensedLocalProvider(VideoProvider):
    name = "licensed_local"
    access_basis = "operator-supplied clips with an explicit licence reference in manifest.csv"

    def __init__(self, root: Path | None = None):
        self.root = root or get_settings().licensed_video_dir

    def enabled(self) -> bool:
        return bool(self.root) and (Path(self.root) / "manifest.csv").exists()

    def discover_game(self, game_pk: int, pitcher_id: int) -> list[VideoCandidate]:
        if not self.enabled():
            return []
        out = []
        with open(Path(self.root) / "manifest.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                if int(r["game_pk"]) != game_pk or not r.get("license_reference"):
                    continue
                f = Path(self.root) / r["file"]
                if not f.exists():
                    continue
                out.append(VideoCandidate(
                    source_provider=self.name, source_event_id=f"{game_pk}-{r['at_bat_number']}-{r['pitch_number']}",
                    game_pk=game_pk, pitcher_id=int(r["pitcher_id"]), batter_id=int(r["batter_id"]),
                    at_bat_number=int(r["at_bat_number"]), pitch_number=int(r["pitch_number"]), inning=None,
                    permitted_media_reference=str(f), playback_method="licensed_local",
                    rights_or_permission_status=f"licensed:{r['license_reference']}",
                ))
        return out

    def check_availability(self, candidate_url: str) -> tuple[str, str]:
        return ("licensed_local", "file present") if Path(candidate_url).exists() else ("unavailable", "file missing")
