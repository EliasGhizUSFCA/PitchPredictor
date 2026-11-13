"""Official MLB pitch-video references via the Statcast playId.

Access basis:
  * Pitch event identifiers (playId) come from the public MLB Stats API live feed for the game.
  * Baseball Savant publishes a public per-pitch video page at /sporty-videos?playId=<id>.
  * PitchIQ only LINKS to that official page (Mode B, external playback). It does not extract,
    download, re-host, or embed the underlying media file, because no embedding/download licence has
    been verified for this integration. Operators who obtain a licence can register clips through the
    LicensedLocalProvider (Mode A) instead.
"""
from __future__ import annotations

import re

from ...http import PoliteClient, SourceUnavailable
from ...ingest import mlb_stats
from .base import VideoCandidate, VideoProvider

SAVANT_VIDEO_PAGE = "https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"


class SavantPlayIdProvider(VideoProvider):
    name = "mlb_savant_playid"
    access_basis = ("playId from MLB Stats API live feed; official public Baseball Savant video page "
                    "linked externally (no download/embedding)")

    def __init__(self, client: PoliteClient | None = None, refetch: bool = False):
        self.client = client or PoliteClient()
        self.refetch = refetch

    def discover_game(self, game_pk: int, pitcher_id: int) -> list[VideoCandidate]:
        feed = mlb_stats.fetch_live_feed(self.client, game_pk, use_cache=not self.refetch)
        out = []
        for fp in mlb_stats.pitches_from_feed(feed, pitcher_id):
            if not fp.play_id:
                continue
            out.append(VideoCandidate(
                source_provider=self.name, source_event_id=fp.play_id, game_pk=game_pk, pitcher_id=fp.pitcher_id,
                batter_id=fp.batter_id, at_bat_number=fp.at_bat_number, pitch_number=fp.pitch_number,
                inning=fp.inning, balls_before=fp.balls_before, strikes_before=fp.strikes_before,
                feed_pitch_type=fp.pitch_type, source_page_url=SAVANT_VIDEO_PAGE.format(play_id=fp.play_id),
                playback_method="external_link", rights_or_permission_status="link_only_public_page",
                event_time=fp.start_time,
            ))
        return out

    def check_availability(self, candidate_url: str) -> tuple[str, str]:
        """Fetch the public page once (rate-limited) and check it presents a video for this play.

        The page title is NOT returned to callers before reveal: it describes the pitch outcome.
        """
        try:
            html = self.client.get(candidate_url, namespace="savant_video_page", suffix=".html")
        except SourceUnavailable as e:
            return ("unavailable", f"page fetch failed: {e}")
        has_video = bool(re.search(r"<video|\.mp4", html))
        if has_video:
            return ("external_available", "official page lists a video for this playId")
        return ("metadata_only", "official page reachable but no video element found")
