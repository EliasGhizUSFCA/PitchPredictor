"""Offline acceptance-test harness.

Runs the REAL ingestion -> features -> training -> evaluation pipeline against the `pitchiq_test`
database using fixtures cut from real Statcast / MLB Stats API responses (tests/make_fixtures.py).
"""
import io
import json
import os
import tempfile
from pathlib import Path

import pytest

TMP = Path(tempfile.mkdtemp(prefix="pitchiq-test-"))
os.environ.setdefault("PITCHIQ_DATABASE_URL", "postgresql+psycopg://pitchiq:pitchiq@localhost:5432/pitchiq_test")
os.environ["PITCHIQ_ARTIFACT_DIR"] = str(TMP / "artifacts")
os.environ["PITCHIQ_DATA_DIR"] = str(TMP / "data")
os.environ["PITCHIQ_HTTP_MIN_INTERVAL_S"] = "0"
# MLflow mirror goes to a throwaway store for the test session (never backend/mlruns).
os.environ["PITCHIQ_MLFLOW_ENABLED"] = "true"
os.environ["PITCHIQ_MLFLOW_TRACKING_URI"] = f"sqlite:///{TMP / 'mlruns' / 'mlflow.db'}"
os.environ["PITCHIQ_MLFLOW_ARTIFACT_ROOT"] = str(TMP / "mlruns" / "artifacts")

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from pitchiq.http import PoliteClient, SourceUnavailable  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
PID = 519242


class FixtureClient(PoliteClient):
    """Serves recorded real responses; unknown URLs raise SourceUnavailable (like an outage)."""

    def __init__(self, fail_games: set[int] | None = None):
        super().__init__(cache_dir=TMP / "raw", min_interval_s=0)
        self.df = pd.read_csv(FIX / "statcast_subset.csv", low_memory=False)
        self.people = json.loads((FIX / "people.json").read_text())["people"]
        self.fail_games = fail_games or set()
        self.calls: list[str] = []

    def get(self, url, params=None, *, namespace, suffix=".txt", use_cache=True, retries=3):
        self.calls.append(url)
        if "statcast_search" in url:
            d = self.df
            season = int(params["hfSea"].strip("|"))
            m = (d.game_date > params["game_date_gt"]) | (d.game_date == params["game_date_gt"])
            m &= (d.game_date < params["game_date_lt"]) | (d.game_date == params["game_date_lt"])
            m &= d.game_year == season
            buf = io.StringIO()
            d[m].to_csv(buf, index=False)
            return buf.getvalue() if m.any() else ""
        if "/feed/live" in url:
            gpk = int(url.split("/game/")[1].split("/")[0])
            f = FIX / "live_feed" / f"{gpk}.json"
            if gpk in self.fail_games or not f.exists():
                raise SourceUnavailable(f"fixture outage for {gpk}")
            return f.read_text()
        if "/v1/people" in url:
            if params and "personIds" in params:
                ids = {int(x) for x in params["personIds"].split(",")}
                return json.dumps({"people": [p for p in self.people if p["id"] in ids]})
            pid = int(url.rstrip("/").split("/")[-1])
            return json.dumps({"people": [p for p in self.people if p["id"] == pid]})
        if "sporty-videos" in url:
            return "<html><video src='x.mp4'></video></html>"
        raise SourceUnavailable(f"no fixture for {url}")


@pytest.fixture(scope="session")
def db_url():
    return os.environ["PITCHIQ_DATABASE_URL"]


@pytest.fixture(scope="session")
def pipeline(db_url):
    """Fresh schema via Alembic migrations, then the full pipeline on fixture data."""
    from alembic import command
    from alembic.config import Config

    from pitchiq.db import get_engine, session_scope
    eng = get_engine(db_url)
    with eng.begin() as c:
        c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
    cfg = Config(str(Path(__file__).parents[1] / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(__file__).parents[1] / "alembic"))
    command.upgrade(cfg, "head")

    from pitchiq.features.store import build_and_store
    from pitchiq.ingest import pipeline as ing
    from pitchiq.ml import train
    from pitchiq.ml.evaluate import benchmark_all
    from pitchiq.video.discovery import build_video_library
    from pitchiq.video.providers.savant import SavantPlayIdProvider

    train.GRIDS.update({"frequency": [{"alpha": 20.0}], "logreg": [{"C": 0.1}],
                        "lightgbm": [{"num_leaves": 7, "min_child_samples": 20, "n_estimators": 200}]})
    client = FixtureClient()
    out = {}
    with session_scope(db_url) as s:
        out["ingest"] = ing.ingest_statcast(s, PID, [2023, 2024, 2025, 2026], client=client)
        out["feeds"] = ing.ingest_game_feeds(s, PID, client=client)
        out["validation"] = ing.validate(s, PID)
        out["features"] = build_and_store(s, PID)
        out["video"] = build_video_library(s, PID, providers=[SavantPlayIdProvider(client)])
    with session_scope(db_url) as s:
        out["models"] = [e.model_version for e in train.train_all(s, PID)]
    with session_scope(db_url) as s:
        out["benchmarks"] = [r.evaluation_run_id for r in benchmark_all(s, PID)]
    out["client"] = client
    return out


@pytest.fixture()
def session(pipeline, db_url):
    from pitchiq.db import get_sessionmaker
    s = get_sessionmaker(db_url)()
    yield s
    s.rollback()
    s.close()


@pytest.fixture()
def api(pipeline):
    from fastapi.testclient import TestClient

    from pitchiq.api.main import app
    return TestClient(app)
