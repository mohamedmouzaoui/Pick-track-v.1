# Pick & Track Agents

A small, explainable demo inspired by the *Pick & Track* thesis (shoplifting / pick-up detection),
with a **multi-agent AI layer** on top of the vision events.

```
 frames ──► detector ──► risk score ──► agents ──► dashboard
 (sim or    PICK /        0-100,         Observer → Analyst
  YOLOv8)   CONCEAL /     every point    Skeptic  → Responder
            CHECKOUT /    has a reason   (LLM optional, rules fallback)
            EXIT
```

## What it does
1. **Vision layer** - for each tracked person it uses the bounding box and wrist keypoints
   (what YOLOv8-pose + ByteTrack give you) and a few zone rules:
   a hand held in a shelf zone then leaving = `PICK`, a hand moving to the torso/pocket area right
   after = `CONCEAL`, time at the checkout = `CHECKOUT`, crossing the exit = `EXIT`.
2. **Risk score** - transparent 0-100 score (picks, concealments, leaving without checkout, paying).
3. **Agents** - run only for people who picked something (cheap):
   - **Observer** writes a factual timeline
   - **Analyst** gives a verdict: `normal` / `suspicious` / `likely_theft`
   - **Skeptic** argues the innocent explanations (fights false positives)
   - **Responder** picks an action: `log_only`, `notify_staff_assist`, `escalate_to_security_review`
4. **Guardrails** - the LLM can move the Analyst verdict at most one step from the rules, bad/empty
   LLM output falls back to rules, the Responder's *decision* is deterministic (the LLM only writes
   the wording), and nothing ever accuses anybody: every alert needs **human review**.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python run_demo.py --scenario mixed        # CLI, fully offline
uvicorn picktrack.api:app --reload         # dashboard on http://127.0.0.1:8000
pytest                                     # 7 tests
```
Scenarios: `mixed`, `honest`, `shoplifter`, `browser`.
Docker: `docker compose up --build`.

## Turn on the LLM agents (optional)
```bash
cp .env.example .env     # then set ONE of:
# GROQ_API_KEY=...        (llama-3.3-70b-versatile by default)
# ANTHROPIC_API_KEY=...   (claude-sonnet-4-6 by default)
```
The dashboard shows an `llm` / `rules` tag on every agent message, so you can see what is real AI.

## Use a real video
```bash
pip install ultralytics
python run_demo.py --video store.mp4
```
Edit `config/zones.json` (rectangles in pixels of a reference frame size; they are rescaled
automatically). `YoloSource` follows the Ultralytics API but was not run in the build sandbox
(no GPU / model weights), so expect to tune it on your own footage.

## Project layout
```
picktrack/models.py     data classes          picktrack/agents.py    the 4 agents + orchestrator
picktrack/zones.py      zone rectangles       picktrack/llm.py       Groq / Anthropic / offline client
picktrack/detector.py   event rules           picktrack/pipeline.py  glue + live triggers
picktrack/risk.py       risk score            picktrack/api.py       FastAPI + static dashboard
picktrack/sources.py    simulator + YOLO      tests/                 pytest suite
```

## Honest limitations
- The simulator replaces real video; the detector is rule-based and sees *hand gestures*, not
  actual products. A real deployment needs object-level detection and lots of false-positive testing.
- Hand-near-torso is a weak signal on its own (phones, wallets, own bags) - hence the Skeptic agent.
- Camera analytics of people has legal / privacy requirements (signage, retention, GDPR-like rules).
  This is a learning and portfolio demo, not a production loss-prevention system.

## CV line (suggestion)
> **Pick & Track Agents** - Python/FastAPI demo: pose-based pick/conceal detection, explainable risk
> scoring and a 4-agent LLM review layer (Observer, Analyst, Skeptic, Responder) with guardrails
> and human-in-the-loop escalation; Dockerized, 7 automated tests.
