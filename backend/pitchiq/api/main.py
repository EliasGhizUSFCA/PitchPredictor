"""PitchIQ FastAPI service. Run: uvicorn pitchiq.api.main:app --reload"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..http import SourceUnavailable
from ..logging_setup import configure_logging
from .routers import challenge, jobs, models, pitchers, simulator, videos

configure_logging()
app = FastAPI(title="PitchIQ AI", version="0.1.0",
              description="MLB next-pitch prediction research API. All probabilities are genuine model outputs.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(LookupError)
async def _lookup(_: Request, e: LookupError):
    return JSONResponse(status_code=404, content={"detail": str(e)})


@app.exception_handler(ValueError)
async def _value(_: Request, e: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(e)})


@app.exception_handler(SourceUnavailable)
async def _source(_: Request, e: SourceUnavailable):
    return JSONResponse(status_code=503, content={"detail": f"Upstream source unavailable: {e}"})


@app.get("/api/health")
def health():
    from sqlalchemy import text

    from ..db import get_engine
    try:
        with get_engine().connect() as c:
            c.execute(text("select 1"))
        db = "ok"
    except Exception as e:  # noqa: BLE001
        db = f"error: {e}"
    return {"status": "ok" if db == "ok" else "degraded", "database": db}


for r in (pitchers.router, simulator.router, models.router, challenge.router, videos.router, jobs.router):
    app.include_router(r, prefix="/api")

from .routers import lifecycle as _lifecycle  # noqa: E402  (np-refit: protocol, gates, rollback, prospective)

app.include_router(_lifecycle.router, prefix="/api")

from .routers import aging as _aging  # noqa: E402  (aging dataset)

app.include_router(_aging.router, prefix="/api")
