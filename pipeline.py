"""Glue: frames -> detector -> risk -> (agents) -> callbacks."""
from __future__ import annotations

import time
from typing import Callable, Optional

from .agents import Case, Orchestrator
from .detector import DetectorConfig, EventDetector
from .models import Event, FrameData
from .risk import HIGH, RiskResult, score_track
from .zones import ZoneSet


class Pipeline:
    def __init__(self, source, zones: ZoneSet, orchestrator: Orchestrator | None = None,
                 config: DetectorConfig | None = None, speed: float = 0.0,
                 on_event: Optional[Callable[[Event], None]] = None,
                 on_frame: Optional[Callable[[FrameData, dict[int, RiskResult]], None]] = None,
                 on_case: Optional[Callable[[Case], None]] = None):
        self.source, self.base_zones = source, zones
        self.orchestrator = orchestrator or Orchestrator()
        self.config, self.speed = config, speed
        self.on_event = on_event or (lambda e: None)
        self.on_frame = on_frame or (lambda f, r: None)
        self.on_case = on_case or (lambda c: None)
        self.detector: EventDetector | None = None
        self.events: list[Event] = []
        self.cases: list[Case] = []
        self._investigated: set[int] = set()

    def run(self) -> list[Case]:
        risks: dict[int, RiskResult] = {}
        for frame in self.source.frames():
            if self.detector is None:
                self.detector = EventDetector(self.base_zones.scaled(frame.width, frame.height), self.config)
            new_events = self.detector.update(frame)
            risks = {tid: score_track(st) for tid, st in self.detector.states.items()}
            for ev in new_events:
                self.events.append(ev)
                self.on_event(ev)
            self.on_frame(frame, risks)
            for tid, st in self.detector.states.items():
                if tid in self._investigated or not (st.picks or st.conceals):
                    continue
                if risks[tid].level == HIGH:
                    self._investigate(tid, risks[tid], "risk_high")
                elif st.exited:
                    self._investigate(tid, risks[tid], "exit")
            if self.speed > 0:
                time.sleep(1 / (getattr(self.source, "fps", 10) * self.speed))
        if self.detector:  # stream ended: close any open case
            for tid, st in self.detector.states.items():
                if tid not in self._investigated and (st.picks or st.conceals):
                    self._investigate(tid, risks[tid], "end_of_stream")
        return self.cases

    def _investigate(self, tid: int, risk: RiskResult, trigger: str) -> None:
        self._investigated.add(tid)
        case = self.orchestrator.investigate(self.detector.states[tid], risk, trigger)
        self.cases.append(case)
        self.on_case(case)
