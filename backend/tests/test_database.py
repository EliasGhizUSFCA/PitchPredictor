"""Database test: real historical pitches can be retrieved and are ordered correctly."""
from sqlalchemy import select

from pitchiq.models import Game, GameLineup, Pitch, PlateAppearance

PID = 519242


def test_pitches_loaded_and_ordered(session, pipeline):
    assert pipeline["ingest"]["pitches"] == 735
    assert pipeline["ingest"]["games"] == 8
    games = session.scalars(select(Game.game_pk)).all()
    for g in games:
        ps = session.scalars(select(Pitch).where(Pitch.game_pk == g, Pitch.pitcher_id == PID)
                             .order_by(Pitch.game_pitch_seq)).all()
        assert [p.game_pitch_seq for p in ps] == list(range(1, len(ps) + 1))
        keys = [(p.at_bat_number, p.pitch_number) for p in ps]
        assert keys == sorted(keys), "game sequence must follow at-bat then pitch-number order"
        # pitch_number restarts each plate appearance; game_pitch_seq never does
        assert max(p.pitch_number for p in ps) < len(ps)


def test_provenance_and_lineups(session, pipeline):
    p = session.scalars(select(Pitch).limit(1)).first()
    assert p.source.startswith("baseball_savant") and p.raw["game_pk"] == p.game_pk
    assert session.scalar(select(GameLineup).limit(1)) is not None
    missing = session.scalars(select(PlateAppearance).where(PlateAppearance.batting_order_slot.is_(None))).all()
    assert missing == []


def test_validation_flags_pitch_clock_rows(pipeline):
    v = pipeline["validation"]
    # Fixture games contain three automatic ball/strike rows absent from the live feed's pitch list.
    assert v["missing_pitch_type"] == 3
    assert v["by_type"].get("pitch_count_mismatch") == 3
