"""'Build Chris Sale Video Library' — automated discovery, matching, verification, coverage reporting."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..config import get_settings
from ..http import SourceUnavailable
from ..logging_setup import log
from ..models import Game, Pitch, VideoAsset
from .matching import PitchKey, match
from .providers.base import VideoProvider
from .providers.licensed_local import LicensedLocalProvider
from .providers.savant import SavantPlayIdProvider

logger = logging.getLogger(__name__)


def default_providers() -> list[VideoProvider]:
    provs: list[VideoProvider] = []
    lic = LicensedLocalProvider()
    if lic.enabled():
        provs.append(lic)
    provs.append(SavantPlayIdProvider())
    return provs


def _games_needing_discovery(session: Session, pitcher_id: int, stale_days: int, force: bool) -> list[int]:
    games = session.scalars(select(Game.game_pk).where(
        Game.game_pk.in_(select(Pitch.game_pk).where(Pitch.pitcher_id == pitcher_id))).order_by(Game.game_date)).all()
    if force:
        return list(games)
    cutoff = datetime.now(timezone.utc) - timedelta(days=stale_days)
    fresh = dict(session.execute(
        select(VideoAsset.game_pk, func.count()).where(
            VideoAsset.pitcher_id == pitcher_id, VideoAsset.match_confidence == "exact_verified",
            VideoAsset.discovered_at >= cutoff).group_by(VideoAsset.game_pk)).all())
    n_pitches = dict(session.execute(select(Pitch.game_pk, func.count()).where(
        Pitch.pitcher_id == pitcher_id, Pitch.pitch_type.is_not(None)).group_by(Pitch.game_pk)).all())
    return [g for g in games if fresh.get(g, 0) < n_pitches.get(g, 0)]


def build_video_library(session: Session, pitcher_id: int, providers: list[VideoProvider] | None = None,
                        max_games: int | None = None, force: bool = False,
                        progress: Callable | None = None, should_cancel: Callable[[], bool] | None = None) -> dict:
    s = get_settings()
    providers = providers or default_providers()
    games = _games_needing_discovery(session, pitcher_id, s.video_metadata_stale_days, force)
    if max_games:
        games = games[:max_games]
    stats = {"games_searched": 0, "candidates": 0, "exact_verified": 0, "strong_review": 0, "possible": 0,
             "unmatched_candidates": 0, "failures": [], "providers": [p.name for p in providers]}
    for gi, gpk in enumerate(games):
        if should_cancel and should_cancel():
            stats["cancelled"] = True
            break
        pitches = session.scalars(select(Pitch).where(Pitch.game_pk == gpk, Pitch.pitcher_id == pitcher_id)).all()
        keys = {(p.at_bat_number, p.pitch_number): PitchKey(p.pitch_uid, p.game_pk, p.pitcher_id, p.batter_id,
                                                            p.at_bat_number, p.pitch_number, p.inning, p.balls,
                                                            p.strikes, p.pitch_type) for p in pitches}
        for prov in providers:
            try:
                cands = prov.discover_game(gpk, pitcher_id)
            except SourceUnavailable as e:
                stats["failures"].append({"game_pk": gpk, "provider": prov.name, "error": str(e)})
                continue
            rows = []
            for c in cands:
                stats["candidates"] += 1
                pk = keys.get((c.at_bat_number, c.pitch_number))
                if pk is None:
                    stats["unmatched_candidates"] += 1
                    conf, ev, uid = "no_match", {"reason": "no Statcast pitch with these keys"}, None
                else:
                    conf, ev = match(pk, c)
                    uid = pk.pitch_uid if conf in ("exact_verified", "strong_review", "possible") else None
                stats[conf] = stats.get(conf, 0) + 1
                rows.append(dict(
                    pitch_uid=uid, source_provider=c.source_provider, source_event_id=c.source_event_id,
                    source_page_url=c.source_page_url, authorized_embed_reference=c.authorized_embed_reference,
                    permitted_media_reference=c.permitted_media_reference, game_pk=gpk, pitcher_id=c.pitcher_id,
                    batter_id=c.batter_id, at_bat_number=c.at_bat_number, pitch_number=c.pitch_number,
                    inning=c.inning, match_confidence=conf, match_evidence=ev,
                    playback_method=c.playback_method, rights_or_permission_status=c.rights_or_permission_status,
                    review_status="auto" if conf == "exact_verified" else "pending",
                    synchronization_status="unaligned" if c.playback_method == "licensed_local" else "not_applicable",
                    video_availability_status="licensed_local" if c.playback_method == "licensed_local" else "unchecked",
                    discovered_at=datetime.now(timezone.utc),
                ))
            if rows:
                stmt = insert(VideoAsset).values(rows)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_video_source_event",
                    set_={k: stmt.excluded[k] for k in ("pitch_uid", "match_confidence", "match_evidence",
                                                        "source_page_url", "discovered_at")})
                session.execute(stmt)
        stats["games_searched"] += 1
        session.flush()
        if progress:
            progress((gi + 1) / max(len(games), 1), f"Game {gi + 1}/{len(games)} ({gpk})", stats["candidates"])
    log(logger, "video_library_built", pitcher_id=pitcher_id, **{k: v for k, v in stats.items() if k != "failures"})
    return stats


def verify_availability(session: Session, pitcher_id: int, limit: int = 50, asset_ids: list[int] | None = None,
                        progress: Callable | None = None, should_cancel: Callable[[], bool] | None = None) -> dict:
    """Check current availability of exact-verified assets (oldest/unchecked first), rate-limited."""
    provs = {p.name: p for p in [SavantPlayIdProvider(), LicensedLocalProvider()]}
    q = select(VideoAsset).where(VideoAsset.pitcher_id == pitcher_id, VideoAsset.match_confidence == "exact_verified")
    if asset_ids:
        q = q.where(VideoAsset.video_asset_id.in_(asset_ids))
    q = q.order_by(VideoAsset.last_verified_at.asc().nulls_first()).limit(limit)
    assets = session.scalars(q).all()
    counts: dict[str, int] = {}
    for i, a in enumerate(assets):
        if should_cancel and should_cancel():
            break
        prov = provs.get(a.source_provider)
        target = a.permitted_media_reference if a.playback_method == "licensed_local" else a.source_page_url
        status, notes = prov.check_availability(target) if prov and target else ("unavailable", "no reference")
        a.video_availability_status, a.verification_notes = status, notes
        a.last_verified_at = datetime.now(timezone.utc)
        counts[status] = counts.get(status, 0) + 1
        if progress:
            progress((i + 1) / max(len(assets), 1), f"Checked {i + 1}/{len(assets)}", i + 1)
    session.flush()
    return {"checked": len(assets), "by_status": counts}


def coverage(session: Session, pitcher_id: int) -> dict:
    total = session.scalar(select(func.count()).select_from(Pitch).where(Pitch.pitcher_id == pitcher_id,
                                                                         Pitch.pitch_type.is_not(None)))
    def cnt(*conds):
        return session.scalar(select(func.count(func.distinct(VideoAsset.pitch_uid))).where(
            VideoAsset.pitcher_id == pitcher_id, VideoAsset.pitch_uid.is_not(None), *conds)) or 0
    verified = VideoAsset.match_confidence == "exact_verified"
    return {
        "total_pitches": total,
        "with_video_reference": cnt(),
        "verified_matches": cnt(verified),
        "in_app_playback": cnt(verified, VideoAsset.playback_method.in_(["official_embed", "licensed_local"]),
                               VideoAsset.video_availability_status.in_(["embed_available", "licensed_local"])),
        "external_viewing": cnt(verified, VideoAsset.playback_method == "external_link",
                                VideoAsset.video_availability_status.in_(["external_available", "unchecked"])),
        "external_confirmed_available": cnt(verified, VideoAsset.video_availability_status == "external_available"),
        "availability_unchecked": cnt(verified, VideoAsset.video_availability_status == "unchecked"),
        "unavailable": cnt(verified, VideoAsset.video_availability_status.in_(["unavailable", "metadata_only"])),
        "awaiting_verification": cnt(VideoAsset.match_confidence.in_(["strong_review", "possible"]),
                                     VideoAsset.review_status == "pending"),
        "last_discovery": session.scalar(select(func.max(VideoAsset.discovered_at)).where(
            VideoAsset.pitcher_id == pitcher_id)),
        "last_verification": session.scalar(select(func.max(VideoAsset.last_verified_at)).where(
            VideoAsset.pitcher_id == pitcher_id)),
    }
