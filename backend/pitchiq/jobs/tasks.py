"""Background task implementations. Each receives (session, params, ctx)."""
from __future__ import annotations

from ..config import get_settings


def _pid(params):
    return int(params.get("pitcher_id") or get_settings().default_pitcher_id)


def t_ingest_statcast(session, params, ctx):
    from ..ingest.pipeline import ingest_statcast
    return ingest_statcast(session, _pid(params), params.get("seasons") or get_settings().seasons, ctx.progress,
                           use_cache=not params.get("refresh", False))


def t_ingest_feeds(session, params, ctx):
    from ..ingest.pipeline import ingest_game_feeds
    return ingest_game_feeds(session, _pid(params), ctx.progress)


def t_validate(session, params, ctx):
    from ..ingest.pipeline import validate
    return validate(session, _pid(params))


def t_build_features(session, params, ctx):
    from ..features.store import build_and_store
    return build_and_store(session, _pid(params))


def t_video_library(session, params, ctx):
    from ..video.discovery import build_video_library
    return build_video_library(session, _pid(params), max_games=params.get("max_games"),
                               force=params.get("force", False), progress=ctx.progress, should_cancel=ctx.cancelled)


def t_video_availability(session, params, ctx):
    from ..video.discovery import verify_availability
    return verify_availability(session, _pid(params), limit=int(params.get("limit", 50)), progress=ctx.progress,
                               should_cancel=ctx.cancelled)


def t_train(session, params, ctx):
    from ..ml.train import train_all
    fams = params.get("families") or ["majority", "frequency", "logreg", "lightgbm"]
    entries = train_all(session, _pid(params), fams, ctx.progress, auto_promote=params.get("auto_promote", True))
    return {"models": [{"version": e.model_version, "family": e.model_family, "status": e.status,
                        "val_log_loss": e.validation_metrics["log_loss"],
                        "val_top1": e.validation_metrics["top1_accuracy"]} for e in entries]}


def t_benchmark(session, params, ctx):
    from ..ml.evaluate import benchmark_all, holdout_benchmark
    if params.get("model_version"):
        runs = [holdout_benchmark(session, _pid(params), params["model_version"], params.get("split", "test"))]
    else:
        runs = benchmark_all(session, _pid(params), params.get("split", "test"), ctx.progress)
    return {"evaluation_runs": [{"id": r.evaluation_run_id, "model_version": r.model_version, "n": r.n_samples,
                                 **{k: r.metrics["summary"][k] for k in ("top1_accuracy", "log_loss", "brier")}}
                                for r in runs]}


def t_walk_forward(session, params, ctx):
    from ..ml.evaluate import walk_forward
    r = walk_forward(session, _pid(params), params.get("family", "lightgbm"), params.get("split", "test"),
                     int(params.get("block_games", 6)), ctx.progress)
    return {"evaluation_run_id": r.evaluation_run_id, **r.metrics["summary"]}


def t_ablation(session, params, ctx):
    from ..ml.evaluate import ablation
    r = ablation(session, _pid(params), params.get("family", "lightgbm"), ctx.progress)
    return {"evaluation_run_id": r.evaluation_run_id, "results": r.metrics["results"]}


def t_mlflow_backfill(session, params, ctx):
    from ..ml.tracking import get_tracker
    return get_tracker().backfill(session, params.get("pitcher_id"))


def t_full_refresh(session, params, ctx):
    out = {}
    steps = [("ingest_statcast", t_ingest_statcast), ("ingest_feeds", t_ingest_feeds), ("validate", t_validate),
             ("build_features", t_build_features), ("video_library", t_video_library)]
    for i, (name, fn) in enumerate(steps):
        if ctx.cancelled():
            out["cancelled_before"] = name
            break
        ctx.note(f"step {name}")
        out[name] = fn(session, params, ctx.sub(i / len(steps), (i + 1) / len(steps)))
        session.commit()
    return out


TASKS = {
    "ingest_statcast": t_ingest_statcast,
    "ingest_feeds": t_ingest_feeds,
    "validate_data": t_validate,
    "build_features": t_build_features,
    "build_video_library": t_video_library,
    "verify_video_availability": t_video_availability,
    "train_models": t_train,
    "benchmark": t_benchmark,
    "walk_forward": t_walk_forward,
    "ablation": t_ablation,
    "full_refresh": t_full_refresh,
    "mlflow_backfill": t_mlflow_backfill,
}

DESCRIPTIONS = {
    "ingest_statcast": "Historical Statcast ingestion (Baseball Savant)",
    "ingest_feeds": "MLB Stats API live feeds: lineups, batter metadata, validation evidence",
    "validate_data": "Automated data validation report",
    "build_features": "Rebuild leakage-safe pre-pitch feature states",
    "build_video_library": "Build Chris Sale Video Library (discover + match official pitch videos)",
    "verify_video_availability": "Check playback availability of verified clips (rate-limited)",
    "train_models": "Train + calibrate all model families (train/validation only)",
    "benchmark": "Full held-out benchmark (locked test split)",
    "walk_forward": "Walk-forward expanding-window evaluation",
    "ablation": "Feature-group ablation on validation",
    "full_refresh": "Ingest -> feeds -> validate -> features -> video library",
    "mlflow_backfill": "Mirror model registry + evaluation runs into MLflow (Postgres stays the source of truth)",
}
