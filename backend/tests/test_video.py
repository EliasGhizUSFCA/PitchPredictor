"""Video discovery, matching, and playback-mode tests."""
import csv

from sqlalchemy import select

from pitchiq.models import VideoAsset
from pitchiq.video.matching import PitchKey, match
from pitchiq.video.providers.base import VideoCandidate
from pitchiq.video.providers.licensed_local import LicensedLocalProvider

PID = 519242


def _key(**kw):
    d = dict(pitch_uid=1, game_pk=10, pitcher_id=PID, batter_id=5, at_bat_number=3, pitch_number=2, inning=1,
             balls=1, strikes=0, pitch_type="SL")
    d.update(kw)
    return PitchKey(**d)


def _cand(**kw):
    d = dict(source_provider="mlb_savant_playid", source_event_id="x", game_pk=10, pitcher_id=PID, batter_id=5,
             at_bat_number=3, pitch_number=2, inning=1, balls_before=1, strikes_before=0, feed_pitch_type="SL")
    d.update(kw)
    return VideoCandidate(**d)


def test_discovery_saves_verified_matches(session, pipeline):
    v = pipeline["video"]
    assert v["games_searched"] == 8 and v["exact_verified"] > 700 and v["failures"] == []
    n = session.query(VideoAsset).filter_by(match_confidence="exact_verified").count()
    assert n == v["exact_verified"]
    a = session.scalars(select(VideoAsset).where(VideoAsset.match_confidence == "exact_verified")).first()
    assert a.playback_method == "external_link" and a.permitted_media_reference is None
    assert a.source_page_url.startswith("https://baseballsavant.mlb.com/sporty-videos?playId=")


def test_discovery_reports_unavailable_source(session, db_url):
    from conftest import FixtureClient
    from pitchiq.video.discovery import build_video_library
    from pitchiq.video.providers.savant import SavantPlayIdProvider
    client = FixtureClient(fail_games={718010})
    stats = build_video_library(session, PID, providers=[SavantPlayIdProvider(client)], force=True)
    assert any(f["game_pk"] == 718010 for f in stats["failures"])
    assert stats["games_searched"] == 8


def test_exact_requires_full_metadata():
    assert match(_key(), _cand())[0] == "exact_verified"
    assert match(_key(), _cand(pitch_number=3))[0] == "possible"           # wrong pitch in the PA
    assert match(_key(), _cand(feed_pitch_type=None))[0] == "strong_review"  # cannot verify classification
    assert match(_key(), _cand(balls_before=2))[0] == "strong_review"     # count disagrees
    assert match(_key(), _cand(batter_id=99, at_bat_number=7))[0] == "no_match"
    # A matching title would not help: titles are not an input to match()
    assert match(_key(), _cand(at_bat_number=4, extra={"title": "Chris Sale strikes out batter"}))[0] != "exact_verified"


def test_licensed_local_provider_and_media_endpoint(session, api, tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    from pitchiq.models import Pitch
    p = session.scalars(select(Pitch).where(Pitch.pitch_type.is_not(None))).first()
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "game_pk", "at_bat_number", "pitch_number", "pitcher_id", "batter_id", "license_reference"])
        w.writerow(["clip.mp4", p.game_pk, p.at_bat_number, p.pitch_number, PID, p.batter_id, "TEST-LICENSE-1"])
        w.writerow(["clip.mp4", p.game_pk, p.at_bat_number, p.pitch_number + 1, PID, p.batter_id, ""])  # no licence
    prov = LicensedLocalProvider(tmp_path)
    cands = prov.discover_game(p.game_pk, PID)
    assert len(cands) == 1 and cands[0].playback_method == "licensed_local"
    from pitchiq.video.discovery import build_video_library
    build_video_library(session, PID, providers=[prov], force=True)
    session.commit()
    a = session.scalars(select(VideoAsset).where(VideoAsset.source_provider == "licensed_local")).first()
    assert a.match_confidence == "exact_verified" and a.rights_or_permission_status == "licensed:TEST-LICENSE-1"
    r = api.get(f"/api/videos/{a.video_asset_id}/media")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    ext = session.scalars(select(VideoAsset).where(VideoAsset.playback_method == "external_link")).first()
    assert api.get(f"/api/videos/{ext.video_asset_id}/media").status_code == 404
    session.delete(a)
    session.commit()


def test_alignment_only_for_controllable_media(api, session):
    ext = session.scalars(select(VideoAsset).where(VideoAsset.playback_method == "external_link")).first()
    r = api.put(f"/api/videos/{ext.video_asset_id}/alignment", json={"pre_pitch_pause_time": 1.2})
    assert r.status_code == 422
