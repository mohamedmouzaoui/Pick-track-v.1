"""Plain data structures shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Point = tuple[float, float]
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2


class Kind:
    """Event kinds emitted by the detector."""

    PICK = "PICK"
    CONCEAL = "CONCEAL"
    CHECKOUT = "CHECKOUT"
    EXIT = "EXIT"


@dataclass
class Person:
    """One tracked person in one frame (what YOLO-pose + a tracker would give you)."""

    track_id: int
    bbox: BBox
    wrists: list[Point] = field(default_factory=list)

    @property
    def feet(self) -> Point:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass
class FrameData:
    t: float  # seconds since the start of the stream
    width: int
    height: int
    persons: list[Person]


@dataclass
class Event:
    t: float
    track_id: int
    kind: str
    zone: Optional[str] = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "t": round(self.t, 2),
            "track_id": self.track_id,
            "kind": self.kind,
            "zone": self.zone,
            "detail": self.detail,
        }


@dataclass
class TrackState:
    """Everything we remember about one person. Counters feed the risk score."""

    track_id: int
    first_seen: float
    last_seen: float
    picks: int = 0
    conceals: int = 0
    checked_out: bool = False
    exited: bool = False
    shelf_time: float = 0.0
    events: list[Event] = field(default_factory=list)
    # transient detector fields
    reach_start: Optional[float] = None
    reach_zone: Optional[str] = None
    pending_pick_t: Optional[float] = None
    last_pick_zone: Optional[str] = None
    conceal_start: Optional[float] = None
    checkout_start: Optional[float] = None
