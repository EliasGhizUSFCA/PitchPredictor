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
    pr = sub.add_parser("promote", help="Gated promotion to production (see `pitchiq gates`)")
    pr.add_argument("model_version")
    pr.add_argument("--force", action="store_true", help="bypass failing gates (recorded as a manual override)")
    rf = sub.add_parser("refit", help="Rolling development selection + final refit on all data before today + gates")
    rf.add_argument("--families", nargs="*", default=["majority", "frequency", "logreg", "lightgbm"])
    rf.add_argument("--no-promote", action="store_true")
    rt = sub.add_parser("retrain", help="Scheduled retraining of the production recipe (job retrain_next_pitch)")
    rt.add_argument("--policy", choices=["after_each_start", "every_6_starts", "weekly", "manual"], default="manual")
    rt.add_argument("--force", action="store_true")
    rt.add_argument("--reselect", action="store_true", help="re-run rolling development selection first")
    rt.add_argument("--no-promote", action="store_true")
    rp = sub.add_parser("replay-policies", help="Historical replay of retraining policies")
    rp.add_argument("--families", nargs="*", default=["frequency", "logreg", "lightgbm"])
    rp.add_argument("--seasons", type=int, nargs="*", default=[2025, 2026])
    gt = sub.add_parser("gates", help="Evaluate + record promotion gates for a model (no status change)")
    gt.add_argument("model_version")
    rb = sub.add_parser("rollback", help="Roll the production model back to previous_production")
    rb.add_argument("model_version")
    sub.add_parser("prospective", help="Record + report the prospective ledger")
    de = sub.add_parser("dev-eval", help="Rolling development evaluation of an existing model's frozen recipe")
    de.add_argument("model_version")
    sub.add_parser("protocol", help="Evaluation roles, rolling-development protocol and gate thresholds")
    mb = sub.add_parser("mlflow-backfill", help="Mirror model_registry + evaluation_runs (archived included) into MLflow")
    mb.add_argument("--orphan-grace-s", type=float, default=3600.0,
                    help="skip orphan tagging for runs newer than this (their transaction may still be open)")
    mu = sub.add_parser("mlflow-ui", help="Start the MLflow UI on the configured tracking store")
    mu.add_argument("--port", type=int, default=5000)
    mu.add_argument("--host", default="127.0.0.1")
    ag = sub.add_parser("aging-etl", help="MLB-wide longitudinal pitcher-season dataset for aging analysis (Savant "
                                          "leaderboards + birth dates) and the target pitcher's career pitches")
    ag.add_argument("--seasons", type=int, nargs="*", help="leaderboard seasons (default PITCHIQ_AGING_SEASONS)")
    ag.add_argument("--career-seasons", type=int, nargs="*",
                    help="history-only pitch-level seasons for --pitcher (default PITCHIQ_CAREER_SEASONS)")
    ag.add_argument("--no-career", action="store_true", help="skip the target pitcher's pitch-level career ingest")
    ag.add_argument("--no-search-pitch-types", action="store_true",
                    help="skip the per-pitch-type grouped Statcast searches (extension / whiffs per pitch type)")
    ag.add_argument("--min-pitches", type=int, help="min pitches for a pitch-type season (low_sample flag)")
    ag.add_argument("--min-season-pitches", type=int, help="min pitches for a qualifying pitcher-season")
    ag.add_argument("--refresh", action="store_true", help="bypass the raw response cache")
    ag.add_argument("--refresh-bio", action="store_true", help="re-fetch birth dates already stored")
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
        command.upgrade(cfg, "heads")
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
            from .ml.refit import after_ingest
            out(build_and_store(s, pid))
            out(after_ingest(s, pid))  # prospective ledger, then the automatic retraining-policy check
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
        elif args.cmd == "aging-etl":
            from .aging.etl import run_aging_etl
            r = run_aging_etl(s, args.seasons, use_cache=not args.refresh,
                              search_pitch_types=not args.no_search_pitch_types,
                              career_pitcher_id=None if args.no_career else pid, career_seasons=args.career_seasons,
                              refresh_bio=args.refresh_bio, min_pitches=args.min_pitches,
                              min_season_pitches=args.min_season_pitches)
            out({k: v for k, v in r.items() if k != "sources"})
        elif args.cmd == "promote":
            from .ml.lifecycle import promote_with_gates
            out(promote_with_gates(s, args.model_version, source="manual: pitchiq promote", force=args.force))
        elif args.cmd == "refit":
            from .ml.refit import refit_all
            out(refit_all(s, pid, args.families, promote=not args.no_promote))
        elif args.cmd == "retrain":
            from .ml.refit import retrain
            out(retrain(s, pid, args.policy, force=args.force, reselect=args.reselect, promote=not args.no_promote))
        elif args.cmd == "replay-policies":
            from .ml.replay import replay_policies
            for r in replay_policies(s, pid, args.families, args.seasons):
                out({"evaluation_run_id": r.evaluation_run_id, "family": r.metrics["family"],
                     "season": r.metrics["season"], "role": r.metrics["evaluation_label"],
                     "policies": {p: {k: d[k] for k in ("log_loss", "brier", "top1_accuracy", "n_refits")}
                                  for p, d in r.metrics["policies"].items()}})
        elif args.cmd == "gates":
            from .ml.lifecycle import check_gates
            out(check_gates(s, args.model_version))
        elif args.cmd == "rollback":
            from .ml.lifecycle import rollback
            out(rollback(s, args.model_version, source="manual: pitchiq rollback"))
        elif args.cmd == "prospective":
            from .ml.prospective import prospective_metrics, record_prospective
            out({"recorded": record_prospective(s, pid), "metrics": prospective_metrics(s, pid)})
        elif args.cmd == "dev-eval":
            from .ml.refit import evaluate_recipe_on_dev
            from .ml.registry import get_entry
            r = evaluate_recipe_on_dev(s, get_entry(s, pid, args.model_version))
            out({"evaluation_run_id": r.evaluation_run_id, "dev_metrics": r.metrics["dev_metrics"]})
        elif args.cmd == "protocol":
            from .ml.protocol import protocol_dict
            out(protocol_dict())


if __name__ == "__main__":
    main()
