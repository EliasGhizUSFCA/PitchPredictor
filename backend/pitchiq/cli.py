"""PitchIQ command line. `pitchiq --help`"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import BACKEND_ROOT, get_settings
from .logging_setup import configure_logging


def _p(args):
    return args.pitcher or get_settings().default_pitcher_id


def main(argv=None):
    configure_logging()
    ap = argparse.ArgumentParser(prog="pitchiq")
    ap.add_argument("--pitcher", type=int, default=None, help="MLBAM pitcher id (default: Chris Sale 519242)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db", help="Apply Alembic migrations")
    i = sub.add_parser("ingest", help="Statcast ingestion")
    i.add_argument("--seasons", type=int, nargs="*")
    i.add_argument("--refresh", action="store_true", help="bypass the raw response cache")
    sub.add_parser("feeds", help="MLB Stats API live feeds (lineups, people)")
    sub.add_parser("validate")
    sub.add_parser("features")
    v = sub.add_parser("videos", help="Build Chris Sale Video Library")
    v.add_argument("--max-games", type=int)
    v.add_argument("--force", action="store_true")
    va = sub.add_parser("verify-videos")
    va.add_argument("--limit", type=int, default=50)
    t = sub.add_parser("train")
    t.add_argument("--families", nargs="*", default=["majority", "frequency", "logreg", "lightgbm"])
    b = sub.add_parser("benchmark")
    b.add_argument("--model-version")
    b.add_argument("--split", default="test")
    w = sub.add_parser("walk-forward")
    w.add_argument("--family", default="lightgbm")
    a = sub.add_parser("ablation")
    a.add_argument("--family", default="lightgbm")
    pr = sub.add_parser("promote")
    pr.add_argument("model_version")
    mb = sub.add_parser("mlflow-backfill", help="Mirror model_registry + evaluation_runs (archived included) into MLflow")
    mb.add_argument("--orphan-grace-s", type=float, default=3600.0,
                    help="skip orphan tagging for runs newer than this (their transaction may still be open)")
    mu = sub.add_parser("mlflow-ui", help="Start the MLflow UI on the configured tracking store")
    mu.add_argument("--port", type=int, default=5000)
    mu.add_argument("--host", default="127.0.0.1")
    sub.add_parser("bootstrap", help="Full pipeline: migrate, ingest, feeds, validate, features, videos, train, benchmark")
    sub.add_parser("worker", help="Run the background job worker")
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    from .db import session_scope
    pid = _p(args)

    def out(x):
        print(json.dumps(x, default=str, indent=2))

    if args.cmd == "init-db":
        from alembic import command
        from alembic.config import Config
        cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
        command.upgrade(cfg, "head")
        return
    if args.cmd == "worker":
        from .jobs.worker import main as wmain
        return wmain()
    if args.cmd == "serve":
        import uvicorn
        return uvicorn.run("pitchiq.api.main:app", host=args.host, port=args.port)
    if args.cmd == "mlflow-ui":
        import os
        import shutil
        uri = get_settings().mlflow_uri
        exe = shutil.which("mlflow")
        if exe is None:
            raise SystemExit("mlflow is not installed (pip install -e '.[tracking]')")
        os.execv(exe, [exe, "ui", "--backend-store-uri", uri, "--host", args.host, "--port", str(args.port)])
    if args.cmd == "bootstrap":
        for c in (["init-db"], ["ingest"], ["feeds"], ["validate"], ["features"], ["videos"], ["train"],
                  ["benchmark"], ["walk-forward"], ["ablation"]):
            print(f"== {c[0]}")
            main((["--pitcher", str(pid)] if args.pitcher else []) + c)
        return
    with session_scope() as s:
        if args.cmd == "ingest":
            from .ingest.pipeline import ingest_statcast
            out(ingest_statcast(s, pid, args.seasons or get_settings().seasons, use_cache=not args.refresh))
        elif args.cmd == "feeds":
            from .ingest.pipeline import ingest_game_feeds
            out(ingest_game_feeds(s, pid))
        elif args.cmd == "validate":
            from .ingest.pipeline import validate
            out(validate(s, pid))
        elif args.cmd == "features":
            from .features.store import build_and_store
            out(build_and_store(s, pid))
        elif args.cmd == "videos":
            from .video.discovery import build_video_library, coverage
            out(build_video_library(s, pid, max_games=args.max_games, force=args.force))
            out(coverage(s, pid))
        elif args.cmd == "verify-videos":
            from .video.discovery import verify_availability
            out(verify_availability(s, pid, limit=args.limit))
        elif args.cmd == "train":
            from .ml.train import train_all
            for e in train_all(s, pid, args.families):
                out({"version": e.model_version, "status": e.status, "validation": e.validation_metrics,
                     "calibration": e.calibration})
        elif args.cmd == "benchmark":
            from .ml.evaluate import benchmark_all, holdout_benchmark
            runs = [holdout_benchmark(s, pid, args.model_version, args.split)] if args.model_version \
                else benchmark_all(s, pid, args.split)
            for r in runs:
                out({"evaluation_run_id": r.evaluation_run_id, "model_version": r.model_version,
                     **r.metrics["summary"], "bootstrap": r.metrics["bootstrap"]})
        elif args.cmd == "walk-forward":
            from .ml.evaluate import walk_forward
            r = walk_forward(s, pid, args.family)
            out({"evaluation_run_id": r.evaluation_run_id, **r.metrics["summary"]})
        elif args.cmd == "ablation":
            from .ml.evaluate import ablation
            out(ablation(s, pid, args.family).metrics)
        elif args.cmd == "mlflow-backfill":
            from .ml.tracking import get_tracker
            out(get_tracker().backfill(s, args.pitcher, orphan_grace_s=args.orphan_grace_s))
        elif args.cmd == "promote":
            from .ml.registry import promote
            promote(s, args.model_version, source="manual: pitchiq promote")
            out({"promoted": args.model_version})


if __name__ == "__main__":
    main()
