import time
from pathlib import Path

from fastapi.testclient import TestClient

from picktrack.agents import Orchestrator
from picktrack.llm import LLMClient
from picktrack.pipeline import Pipeline
from picktrack.sources import SimulatedSource
from picktrack.zones import ZoneSet

ZONES = ZoneSet.from_json(Path(__file__).parent.parent / "config" / "zones.json")


def run(scenario, llm=None):
    pipe = Pipeline(SimulatedSource(scenario), ZONES, Orchestrator(llm))
    pipe.run()
    return pipe


def kinds(pipe, tid):
    return [e.kind for e in pipe.events if e.track_id == tid]


def test_honest_shopper_is_low_risk():
    p = run("honest")
    assert kinds(p, 1) == ["PICK", "CHECKOUT", "EXIT"]
    case = p.cases[0]
    assert case.risk["level"] == "LOW"
    assert case.responder["action"] == "log_only"


def test_shoplifter_is_escalated_to_humans():
    p = run("shoplifter")
    assert kinds(p, 2).count("CONCEAL") == 2
    case = p.cases[0]
    assert case.risk["level"] == "HIGH"
    assert case.analyst["verdict"] == "likely_theft"
    assert case.responder["action"] == "escalate_to_security_review"
    assert case.responder["requires_human_review"] is True


def test_quick_touch_is_not_a_pick_and_needs_no_agents():
    p = run("browser")
    assert "PICK" not in kinds(p, 3)
    assert p.cases == []


def test_mixed_tracks_stay_separate():
    p = run("mixed")
    assert {c.track_id for c in p.cases} == {1, 2}
    assert "CHECKOUT" not in kinds(p, 2)


class FakeLLM(LLMClient):
    """Pretends to be a model that overreacts: tries to jump straight to 'likely_theft'."""

    def __init__(self):
        super().__init__("offline")

    @property
    def enabled(self):
        return True

    def complete(self, system, user, max_tokens=400):
        if "loss-prevention analyst" in system:
            return 'Sure! {"verdict": "likely_theft", "confidence": 0.99, "reasoning": "looks bad"}'
        if "skeptical reviewer" in system:
            return '{"doubt": 0.1, "alternatives": ["could be own phone"]}'
        return "Narrative from the model."


def test_llm_guardrail_limits_verdict_jump():
    p = run("honest", FakeLLM())  # rules say 'normal'; LLM wants 'likely_theft'
    case = p.cases[0]
    assert case.analyst["source"] == "llm"
    assert case.analyst["verdict"] == "suspicious"  # clamped: one step max
    assert case.observer["source"] == "llm"


def test_bad_llm_output_falls_back_to_rules():
    class Broken(FakeLLM):
        def complete(self, *a, **k):
            return "not json at all"

    case = run("shoplifter", Broken()).cases[0]
    assert case.analyst["verdict"] == "likely_theft" and case.analyst["source"] == "rules"


def test_api_end_to_end():
    from picktrack.api import app

    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.post("/api/run?scenario=nope").status_code == 400
    assert c.post("/api/run?scenario=shoplifter&speed=0").status_code == 200
    for _ in range(100):
        s = c.get("/api/state").json()
        if s["status"] == "done":
            break
        time.sleep(0.1)
    assert s["status"] == "done"
    assert s["cases"][0]["responder"]["requires_human_review"] is True
    assert c.get("/").status_code == 200
