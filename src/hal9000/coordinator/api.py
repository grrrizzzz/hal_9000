"""HAL's HTTP API, for the scheduler (n8n) and the Home Assistant watchdog.

    POST /runs/evening   {"date": "YYYY-MM-DD"?, "shadow": bool?, "no_llm": bool?}
    POST /runs/replay    {"fixture": "path"}
    GET  /health         last success, status, degraded flag, month-to-date spend
    GET  /cost           month-to-date spend by agent and model

Every endpoint except /health needs ``Authorization: Bearer $HAL_API_TOKEN``.
Endpoints are plain ``def`` because the agent loop is synchronous; FastAPI runs
them in its thread pool.
"""

from __future__ import annotations

import datetime as dt
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from hal9000 import config
from hal9000.coordinator.runner import RunOptions, open_database, run_evening

__all__ = ["app"]

app = FastAPI(title="HAL 9000", version="0.2.0")


def _auth(authorization: str = Header(default="")) -> None:
    token = config.settings().api_token
    if not token:
        raise HTTPException(status_code=503, detail="HAL_API_TOKEN is not set on the server")
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="missing or wrong bearer token")


class EveningRequest(BaseModel):
    date: dt.date | None = None
    shadow: bool = False
    no_llm: bool = False


class ReplayRequest(BaseModel):
    fixture: str
    no_llm: bool = True


@app.post("/runs/evening", dependencies=[Depends(_auth)])
def evening(body: EveningRequest | None = None) -> dict:
    body = body or EveningRequest()
    try:
        result = run_evening(RunOptions(day=body.date, shadow=body.shadow, no_llm=body.no_llm))
    except config.ConfigError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not result.report_delivered:
        raise HTTPException(status_code=500, detail=result.model_dump())
    return result.model_dump()


@app.post("/runs/replay", dependencies=[Depends(_auth)])
def replay(body: ReplayRequest) -> dict:
    result = run_evening(RunOptions(replay=body.fixture, no_llm=body.no_llm, announce=False))
    return result.model_dump()


@app.get("/health")
def health() -> dict:
    settings = config.settings()
    path = settings.data_dir / "health.json"
    if path.exists():
        return json.loads(path.read_text())
    db = open_database(settings)
    return {"last_success": db.last_success(), "last_status": None, "degraded": False}


@app.get("/cost", dependencies=[Depends(_auth)])
def cost() -> dict:
    settings = config.settings()
    today = datetime.now(ZoneInfo(settings.timezone)).date()
    db = open_database(settings)
    return {
        "month": today.strftime("%Y-%m"),
        "month_to_date_usd": db.month_cost(today.year, today.month),
        "cap_usd": settings.monthly_cap_usd,
        "breakdown": db.cost_breakdown(today.year, today.month),
    }
