"""Frame sources.

SimulatedSource  - scripted people walking around a 1280x720 store. No video/GPU needed.
YoloSource       - real video via Ultralytics YOLOv8-pose + ByteTrack (optional install).
"""
from __future__ import annotations

import math
import random
from typing import Iterator

from .models import FrameData, Person

W, H, FPS = 1280, 720, 10
BOX_W, BOX_H = 90, 220
SPEED = 150  # px / second

ENTRANCE = (60, 520)
STAND = {"A": (300, 330), "B": (900, 330)}  # where people stand in front of a shelf
CHECKOUT_STAND = (650, 560)
EXIT_STAND = (1200, 560)
SHELF_REACH_Y = 240  # y of a wrist reaching into a shelf

# Each step: ("walk", (x, y)) | ("reach", seconds) | ("conceal", seconds) | ("wait", seconds)
SCENARIOS: dict[str, list[tuple[int, float, list[tuple]]]] = {
    # (track_id, start_time_s, steps)
    "honest": [
        (1, 0.0, [("walk", STAND["A"]), ("reach", 1.2), ("wait", 0.5),
                  ("walk", CHECKOUT_STAND), ("wait", 3.0), ("walk", EXIT_STAND)]),
    ],
    "shoplifter": [
        (2, 0.0, [("walk", STAND["B"]), ("reach", 1.0), ("conceal", 1.5),
                  ("reach", 1.0), ("conceal", 1.5), ("walk", EXIT_STAND)]),
    ],
    "browser": [
        (3, 0.0, [("walk", STAND["A"]), ("reach", 0.2), ("wait", 4.0), ("reach", 0.2),
                  ("walk", (1000, 330)), ("walk", EXIT_STAND)]),
    ],
    "mixed": [
        (1, 0.0, [("walk", STAND["A"]), ("reach", 1.2), ("wait", 0.5),
                  ("walk", CHECKOUT_STAND), ("wait", 3.0), ("walk", EXIT_STAND)]),
        (2, 3.0, [("walk", STAND["B"]), ("reach", 1.0), ("conceal", 1.5),
                  ("reach", 1.0), ("conceal", 1.5), ("walk", EXIT_STAND)]),
        (3, 6.0, [("walk", STAND["A"]), ("reach", 0.2), ("wait", 4.0), ("reach", 0.2),
                  ("walk", (1000, 330)), ("walk", EXIT_STAND)]),
    ],
}


def _person(track_id: int, pos: tuple[float, float], mode: str, rng: random.Random) -> Person:
    cx, cy = pos
    j = lambda: rng.uniform(-3, 3)  # noqa: E731 - tiny sensor noise
    bbox = (cx - BOX_W / 2, cy - BOX_H / 2, cx + BOX_W / 2, cy + BOX_H / 2)
    side_l = (cx - 50 + j(), cy + 30 + j())
    if mode == "reach":
        wrists = [(cx + j(), SHELF_REACH_Y + j()), side_l]
    elif mode == "conceal":
        wrists = [(cx + j(), cy + j()), side_l]  # hand moves to the torso / pocket area
    else:
        wrists = [(cx + 50 + j(), cy + 30 + j()), side_l]
    return Person(track_id, bbox, wrists)


def _render_actor(track_id: int, steps: list[tuple], rng: random.Random) -> list[Person]:
    pos, out = ENTRANCE, []
    for step in steps:
        if step[0] == "walk":
            tx, ty = step[1]
            n = max(1, round(math.dist(pos, (tx, ty)) / SPEED * FPS))
            for i in range(1, n + 1):
                f = i / n
                p = (pos[0] + (tx - pos[0]) * f, pos[1] + (ty - pos[1]) * f)
                out.append(_person(track_id, p, "walk", rng))
            pos = (tx, ty)
        else:
            for _ in range(max(1, round(step[1] * FPS))):
                out.append(_person(track_id, pos, step[0] if step[0] != "wait" else "idle", rng))
    return out


class SimulatedSource:
    fps = FPS

    def __init__(self, scenario: str = "mixed", seed: int = 7):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
        self.scenario, self.seed = scenario, seed

    def frames(self) -> Iterator[FrameData]:
        rng = random.Random(self.seed)
        timelines = [
            (round(start * FPS), _render_actor(tid, steps, rng))
            for tid, start, steps in SCENARIOS[self.scenario]
        ]
        total = max(s + len(tl) for s, tl in timelines)
        for i in range(total):
            persons = [tl[i - s] for s, tl in timelines if 0 <= i - s < len(tl)]
            yield FrameData(i / FPS, W, H, persons)


class YoloSource:
    """Real video: `pip install ultralytics` (pulls PyTorch). Not used by the tests.

    Keypoints 9 and 10 are the left / right wrists in the COCO-17 layout.
    """

    def __init__(self, video: str, model: str = "yolov8n-pose.pt", conf: float = 0.4):
        self.video, self.model_name, self.conf = video, model, conf
        self.fps = 25.0

    def frames(self) -> Iterator[FrameData]:
        import cv2  # noqa: WPS433 - lazy: heavy optional dependency
        from ultralytics import YOLO

        cap = cv2.VideoCapture(self.video)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        cap.release()
        model = YOLO(self.model_name)
        results = model.track(self.video, persist=True, stream=True, conf=self.conf,
                              tracker="bytetrack.yaml", classes=[0], verbose=False)
        for i, r in enumerate(results):
            h, w = r.orig_shape
            persons = []
            if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                ids = r.boxes.id.int().tolist()
                boxes = r.boxes.xyxy.tolist()
                kps = r.keypoints.xy.tolist()
                kconf = r.keypoints.conf.tolist() if r.keypoints.conf is not None else None
                for n, (tid, box, kp) in enumerate(zip(ids, boxes, kps)):
                    wrists = [
                        tuple(kp[k]) for k in (9, 10)
                        if len(kp) > k and (kconf is None or kconf[n][k] > 0.3) and kp[k][0] > 0
                    ]
                    persons.append(Person(tid, tuple(box), wrists))
            yield FrameData(i / self.fps, w, h, persons)
