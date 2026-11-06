"""CLI demo.  python run_demo.py --scenario mixed
           python run_demo.py --video store.mp4   (needs: pip install ultralytics)
"""
import argparse
from pathlib import Path

import picktrack  # noqa: F401  (loads .env)
from picktrack.agents import Orchestrator
from picktrack.llm import LLMClient
from picktrack.pipeline import Pipeline
from picktrack.sources import SCENARIOS, SimulatedSource, YoloSource
from picktrack.zones import ZoneSet

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick & Track Agents demo")
    ap.add_argument("--scenario", default="mixed", choices=sorted(SCENARIOS))
    ap.add_argument("--video", help="path to a video file (uses YOLOv8-pose instead of the simulator)")
    ap.add_argument("--zones", default=str(ROOT / "config" / "zones.json"))
    ap.add_argument("--speed", type=float, default=0.0, help="0 = as fast as possible, 1 = real time")
    args = ap.parse_args()

    llm = LLMClient.from_env()
    print(f"LLM provider: {llm.provider} ({llm.model})\n")
    source = YoloSource(args.video) if args.video else SimulatedSource(args.scenario)
    pipe = Pipeline(
        source, ZoneSet.from_json(args.zones), Orchestrator(llm), speed=args.speed,
        on_event=lambda e: print(f"[{e.t:6.1f}s] #{e.track_id} {e.kind:<8} {e.zone or ''} {e.detail or ''}"),
    )
    cases = pipe.run()

    print("\n" + "=" * 70)
    for c in cases:
        print(f"CASE {c.case_id} | person #{c.track_id} | {c.risk['level']} {c.risk['score']}/100 "
              f"| trigger: {c.trigger}")
        print(f"  Observer : {c.observer['narrative']}")
        print(f"  Analyst  : {c.analyst['verdict']} ({c.analyst['confidence']}) - {c.analyst['reasoning']}")
        print(f"  Skeptic  : doubt={c.skeptic['doubt']}")
        for alt in c.skeptic["alternatives"]:
            print(f"             - {alt}")
        print(f"  Responder: {c.responder['action']} (human review: {c.responder['requires_human_review']})")
        print(f"             {c.responder['message']}\n")
    if llm.last_error:
        print(f"(LLM fell back to rules at least once: {llm.last_error})")


if __name__ == "__main__":
    main()
