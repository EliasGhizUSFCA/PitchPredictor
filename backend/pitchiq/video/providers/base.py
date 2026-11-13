"""Video provider adapter interface.

A provider turns (game, pitch records) into `VideoCandidate`s carrying the metadata needed to verify a
match, plus the playback method its rights allow. Providers must only use access methods the source
permits; anything else is out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VideoCandidate:
    source_provider: str
    source_event_id: str
    game_pk: int
    pitcher_id: int | None
    batter_id: int | None
    at_bat_number: int | None
    pitch_number: int | None
    inning: int | None
    balls_before: int | None = None
    strikes_before: int | None = None
    feed_pitch_type: str | None = None  # used ONLY for match verification, never as a model input
    source_page_url: str | None = None
    authorized_embed_reference: str | None = None
    permitted_media_reference: str | None = None
    playback_method: str = "external_link"  # external_link | official_embed | licensed_local
    rights_or_permission_status: str = "link_only_public_page"
    event_time: str | None = None
    extra: dict = field(default_factory=dict)


class VideoProvider:
    name = "base"
    #: Human-readable description of the access method and its permission basis.
    access_basis = ""

    def discover_game(self, game_pk: int, pitcher_id: int) -> list[VideoCandidate]:
        raise NotImplementedError

    def check_availability(self, candidate_url: str) -> tuple[str, str]:
        """Return (availability_status, notes)."""
        return ("unchecked", "provider does not support availability checks")
