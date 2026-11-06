"""FastAPI dashboard. Run:  uvicorn picktrack.api:app --reload"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

import picktrack  # noqa: F401  (loads .env)
from .agents import Orchestrator
from .llm import LLMClient
from .pipeline import Pipeline
from .sources import SCENARIOS, SimulatedSource
from .zones import ZoneSet

ROOT = Path(__file__).resolve().parent.parent
ZONES = ZoneSet.from_json(ROOT / "config" / "zones.json")

app = FastAPI(title="Pick & Track Agents", version=picktrack.__version__)


class Runner:
    """Runs one pipeline at a time in a background thread and exposes a snapshot for the UI."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.llm = LLMClient.from_env()
        self.thread: threading.Thread | None = None
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.status, self.scenario, self.t = "idle", None, 0.0
            self.events, self.cases, self.persons, self.risks = [], [], [], {}

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, scenario: str, speed: float) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("already running")
            self.reset()
            self.status, self.scenario = "running", scenario
        pipe = Pipeline(
            SimulatedSource(scenario), ZONES, Orchestrator(self.llm), speed=speed,
            on_event=self._on_event, on_frame=self._on_frame, on_case=self._on_case,
        )
        self.thread = threading.Thread(target=self._run, args=(pipe,), daemon=True)
        self.thread.start()

    def _run(self, pipe: Pipeline) -> None:
        try:
            pipe.run()
            status = "done"
        except Exception as exc:  # surface the error in the UI instead of dying silently
            status = f"error: {exc}"
        with self.lock:
            self.status = status

    def _on_event(self, e) -> None:
        with self.lock:
            self.events.append(e.to_dict())

    def _on_case(self, c) -> None:
        with self.lock:
            self.cases.append(c.to_dict())

    def _on_frame(self, frame, risks) -> None:
        with self.lock:
            self.t = round(frame.t, 1)
            self.risks = {tid: r.to_dict() for tid, r in risks.items()}
            self.persons = [
                {"id": p.track_id, "bbox": list(p.bbox), "wrists": [list(w) for w in p.wrists],
                 "level": risks[p.track_id].level if p.track_id in risks else "LOW"}
                for p in frame.persons
            ]

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status, "scenario": self.scenario, "t": self.t,
                "llm_provider": self.llm.provider, "llm_model": self.llm.model,
                "zones": ZONES.to_dict(), "persons": self.persons, "risks": self.risks,
                "events": list(self.events), "cases": list(self.cases),
            }


runner = Runner()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": runner.llm.provider, "scenarios": sorted(SCENARIOS)}


@app.post("/api/run")
def run(scenario: str = "mixed", speed: float = 8.0) -> dict:
    if scenario not in SCENARIOS:
        raise HTTPException(400, f"unknown scenario; choose from {sorted(SCENARIOS)}")
    try:
        runner.start(scenario, max(0.0, min(speed, 100.0)))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"started": scenario}


@app.get("/api/state")
def state() -> dict:
    return runner.snapshot()


@app.post("/api/reset")
def reset() -> dict:
    if runner.running:
        raise HTTPException(409, "a run is in progress")
    runner.reset()
    return {"status": "idle"}
