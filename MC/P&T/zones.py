"""Rectangular store zones (shelf / checkout / exit), defined in config/zones.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import Point


@dataclass(frozen=True)
class Zone:
    name: str
    kind: str  # "shelf" | "checkout" | "exit"
    rect: tuple[float, float, float, float]

    def contains(self, p: Point) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= p[0] <= x2 and y1 <= p[1] <= y2


class ZoneSet:
    def __init__(self, zones: list[Zone], frame_size: tuple[int, int]):
        self.zones = zones
        self.frame_size = frame_size

    @classmethod
    def from_json(cls, path: str | Path) -> "ZoneSet":
        data = json.loads(Path(path).read_text())
        zones = [Zone(z["name"], z["kind"], tuple(z["rect"])) for z in data["zones"]]
        return cls(zones, tuple(data["frame_size"]))

    def scaled(self, width: int, height: int) -> "ZoneSet":
        """Rescale zones from the reference frame size to the real stream size."""
        sx, sy = width / self.frame_size[0], height / self.frame_size[1]
        zs = [
            Zone(z.name, z.kind, (z.rect[0] * sx, z.rect[1] * sy, z.rect[2] * sx, z.rect[3] * sy))
            for z in self.zones
        ]
        return ZoneSet(zs, (width, height))

    def hit(self, point: Point, kind: str) -> Zone | None:
        for z in self.zones:
            if z.kind == kind and z.contains(point):
                return z
        return None

    def to_dict(self) -> dict:
        return {
            "frame_size": list(self.frame_size),
            "zones": [{"name": z.name, "kind": z.kind, "rect": list(z.rect)} for z in self.zones],
        }
