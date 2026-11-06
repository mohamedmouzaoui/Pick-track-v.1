"""A small multi-agent layer on top of the vision events.

Observer  -> writes a factual timeline from the raw events
Analyst   -> gives a verdict (normal / suspicious / likely_theft)
Skeptic   -> argues the innocent explanations (devil's advocate against false positives)
Responder -> picks a *human-in-the-loop* action; it never accuses anybody

Each agent asks the LLM when one is configured and otherwise uses deterministic rules,
so the demo runs offline. The agents share a Case "blackboard" (plain dicts).
Guardrail: the LLM may move the Analyst verdict at most one step away from the rules.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

from .llm import LLMClient
from .models import Event, Kind, TrackState
from .risk import HIGH, LOW, MEDIUM, RiskResult

VERDICTS = ["normal", "suspicious", "likely_theft"]
BASELINE = {LOW: "normal", MEDIUM: "suspicious", HIGH: "likely_theft"}


def parse_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


class Agent:
    name = "agent"
    system = ""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def ask(self, payload: dict, max_tokens: int = 400) -> Optional[str]:
        return self.llm.complete(self.system, json.dumps(payload, ensure_ascii=False), max_tokens)


# --------------------------------------------------------------------------- Observer
class ObserverAgent(Agent):
    name = "observer"
    system = (
        "You are a retail video-analytics narrator. You receive JSON events about ONE anonymous "
        "tracked person. Write 2-3 neutral, factual sentences describing the timeline. Use only "
        "the facts given. Do not speculate about intent, identity, or guilt."
    )

    @staticmethod
    def _phrase(e: Event) -> str:
        if e.kind == Kind.PICK:
            return f"at {e.t:.1f}s reached into {e.zone} for {e.detail.get('reach_s')}s and took an item"
        if e.kind == Kind.CONCEAL:
            return f"at {e.t:.1f}s moved a hand to the torso/pocket area after leaving {e.zone}"
        if e.kind == Kind.CHECKOUT:
            return f"at {e.t:.1f}s spent time at the {e.zone}"
        paid = "after a checkout" if e.detail.get("paid") else "without any checkout event"
        return f"at {e.t:.1f}s left through the {e.zone} {paid}"

    def run(self, track_id: int, events: list[Event]) -> dict:
        payload = {"track_id": track_id, "events": [e.to_dict() for e in events]}
        text = self.ask(payload, 200)
        if text:
            return {"narrative": text.strip(), "source": "llm"}
        parts = "; ".join(self._phrase(e) for e in events) or "no notable events"
        return {"narrative": f"Person #{track_id}: {parts}.", "source": "rules"}


# --------------------------------------------------------------------------- Analyst
class AnalystAgent(Agent):
    name = "analyst"
    system = (
        "You are a loss-prevention analyst. Given a narrative and a rule-based risk score, return "
        'ONLY JSON: {"verdict": "normal|suspicious|likely_theft", "confidence": 0-1, '
        '"reasoning": "<=2 sentences"}. Be conservative: a false accusation is worse than a miss.'
    )

    def run(self, narrative: str, risk: RiskResult) -> dict:
        base = BASELINE[risk.level]
        out = {"verdict": base, "confidence": round(risk.score / 100, 2),
               "reasoning": "; ".join(risk.reasons) or "No risky behaviour detected.", "source": "rules"}
        data = parse_json(self.ask({"narrative": narrative, "risk": risk.to_dict()}, 250))
        if data and data.get("verdict") in VERDICTS:
            # guardrail: LLM can only nudge the verdict by one step from the rules
            lo, hi = VERDICTS.index(base) - 1, VERDICTS.index(base) + 1
            idx = min(max(VERDICTS.index(data["verdict"]), lo), hi)
            try:
                conf = max(0.0, min(1.0, float(data.get("confidence", out["confidence"]))))
            except (TypeError, ValueError):
                conf = out["confidence"]
            out.update(verdict=VERDICTS[idx], confidence=round(conf, 2),
                       reasoning=str(data.get("reasoning", out["reasoning"]))[:400], source="llm")
        return out


# --------------------------------------------------------------------------- Skeptic
class SkepticAgent(Agent):
    name = "skeptic"
    system = (
        "You are a skeptical reviewer defending the customer. Given a narrative, list innocent "
        'explanations. Return ONLY JSON: {"doubt": 0-1, "alternatives": ["...", "..."]} where doubt '
        "is how likely the behaviour is innocent."
    )

    def run(self, narrative: str, st: TrackState) -> dict:
        doubt, alts = 0.2, []
        if st.checked_out:
            doubt += 0.3
            alts.append("The person spent time at the checkout, so the items may have been paid for.")
        if not st.conceals:
            doubt += 0.2
            alts.append("No concealment gesture was seen; items may simply be carried or in a basket.")
        if st.picks == 1:
            doubt += 0.1
            alts.append("Only one item was picked up: a single, low-value pick is common.")
        if st.conceals:
            alts.append("A hand near the torso can mean a phone, wallet or own bag - not store goods.")
        if st.exited and not st.checked_out:
            alts.append("Self-checkout or a staff-assisted payment may be outside the camera's view.")
        out = {"doubt": round(min(doubt, 0.95), 2), "alternatives": alts, "source": "rules"}
        data = parse_json(self.ask({"narrative": narrative, "events": [e.to_dict() for e in st.events]}, 250))
        if data and isinstance(data.get("alternatives"), list):
            try:
                d = max(0.0, min(1.0, float(data.get("doubt", out["doubt"]))))
            except (TypeError, ValueError):
                d = out["doubt"]
            out.update(doubt=round(d, 2), alternatives=[str(a)[:200] for a in data["alternatives"][:4]],
                       source="llm")
        return out


# --------------------------------------------------------------------------- Responder
class ResponderAgent(Agent):
    name = "responder"
    system = (
        "You write one short, polite internal note for store staff (max 2 sentences). Suggest "
        "customer-service-style approaches such as offering help. Never accuse anyone."
    )

    def run(self, track_id: int, risk: RiskResult, analyst: dict, skeptic: dict) -> dict:
        # Deterministic policy: the LLM only writes the wording, never the decision.
        if risk.level == HIGH and analyst["verdict"] == "likely_theft" and skeptic["doubt"] < 0.5:
            action = "escalate_to_security_review"
        elif risk.level in (HIGH, MEDIUM) or analyst["verdict"] != "normal":
            action = "notify_staff_assist"
        else:
            action = "log_only"
        fallback = {
            "escalate_to_security_review": f"Person #{track_id} flagged {risk.level} ({risk.score}/100). "
            "Security should review the footage and, if needed, approach to offer help. Do not accuse.",
            "notify_staff_assist": f"Person #{track_id} flagged {risk.level} ({risk.score}/100). "
            "A staff member could walk by and offer assistance.",
            "log_only": f"Person #{track_id} looks normal ({risk.score}/100). Logged only.",
        }[action]
        text = None
        if action != "log_only":
            text = self.ask({"action": action, "risk": risk.to_dict(), "verdict": analyst["verdict"],
                             "doubt": skeptic["doubt"]}, 120)
        return {"action": action, "requires_human_review": action != "log_only",
                "message": (text or fallback).strip(), "source": "llm" if text else "rules"}


# --------------------------------------------------------------------------- Orchestrator
@dataclass
class Case:
    case_id: int
    track_id: int
    trigger: str
    risk: dict
    observer: dict = field(default_factory=dict)
    analyst: dict = field(default_factory=dict)
    skeptic: dict = field(default_factory=dict)
    responder: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Orchestrator:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient("offline")
        self.observer = ObserverAgent(self.llm)
        self.analyst = AnalystAgent(self.llm)
        self.skeptic = SkepticAgent(self.llm)
        self.responder = ResponderAgent(self.llm)
        self._ids = count(1)

    def investigate(self, st: TrackState, risk: RiskResult, trigger: str) -> Case:
        case = Case(next(self._ids), st.track_id, trigger, risk.to_dict())
        case.observer = self.observer.run(st.track_id, st.events)
        case.analyst = self.analyst.run(case.observer["narrative"], risk)
        case.skeptic = self.skeptic.run(case.observer["narrative"], st)
        case.responder = self.responder.run(st.track_id, risk, case.analyst, case.skeptic)
        return case
