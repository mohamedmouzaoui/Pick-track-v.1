"""Rule-based event detector: turns tracked people + wrist keypoints into store events.

Heuristics (deliberately simple and explainable):
  PICK      a wrist stays inside a shelf zone >= min_reach seconds, then leaves it
  CONCEAL   within conceal_window s after a PICK, a wrist stays in the torso area >= min_conceal s
  CHECKOUT  the person's feet stay in the checkout zone >= min_checkout_dwell s
  EXIT      the person's feet enter the exit zone
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Event, FrameData, Kind, Person, Point, TrackState
from .zones import ZoneSet


@dataclass
class DetectorConfig:
    min_reach: float = 0.4
    conceal_window: float = 6.0
    min_conceal: float = 0.5
    min_checkout_dwell: float = 1.0


def in_torso(person: Person, wrist: Point) -> bool:
    x1, y1, x2, y2 = person.bbox
    w, h = x2 - x1, y2 - y1
    return x1 + 0.15 * w <= wrist[0] <= x2 - 0.15 * w and y1 + 0.35 * h <= wrist[1] <= y1 + 0.75 * h


class EventDetector:
    def __init__(self, zones: ZoneSet, config: DetectorConfig | None = None):
        self.zones = zones
        self.cfg = config or DetectorConfig()
        self.states: dict[int, TrackState] = {}

    def update(self, frame: FrameData) -> list[Event]:
        out: list[Event] = []
        for p in frame.persons:
            st = self.states.get(p.track_id)
            if st is None:
                st = self.states[p.track_id] = TrackState(p.track_id, frame.t, frame.t)
            st.last_seen = frame.t
            out += self._pick(st, p, frame.t)
            out += self._conceal(st, p, frame.t)
            out += self._checkout(st, p, frame.t)
            out += self._exit(st, p, frame.t)
        for ev in out:
            self.states[ev.track_id].events.append(ev)
        return out

    def _pick(self, st: TrackState, p: Person, t: float) -> list[Event]:
        zone = next((z for w in p.wrists if (z := self.zones.hit(w, "shelf"))), None)
        if zone and st.reach_start is None:
            st.reach_start, st.reach_zone = t, zone.name
        elif not zone and st.reach_start is not None:
            dur, name = t - st.reach_start, st.reach_zone
            st.reach_start = st.reach_zone = None
            if dur >= self.cfg.min_reach:
                st.picks += 1
                st.shelf_time += dur
                st.pending_pick_t, st.last_pick_zone, st.conceal_start = t, name, None
                return [Event(t, st.track_id, Kind.PICK, name, {"reach_s": round(dur, 2)})]
        return []

    def _conceal(self, st: TrackState, p: Person, t: float) -> list[Event]:
        if st.pending_pick_t is None:
            return []
        if t - st.pending_pick_t > self.cfg.conceal_window:
            st.pending_pick_t = st.conceal_start = None
            return []
        if any(in_torso(p, w) for w in p.wrists):
            if st.conceal_start is None:
                st.conceal_start = t
            elif t - st.conceal_start >= self.cfg.min_conceal:
                st.conceals += 1
                st.pending_pick_t = st.conceal_start = None
                return [Event(t, st.track_id, Kind.CONCEAL, st.last_pick_zone, {})]
        else:
            st.conceal_start = None
        return []

    def _checkout(self, st: TrackState, p: Person, t: float) -> list[Event]:
        if st.checked_out:
            return []
        zone = self.zones.hit(p.feet, "checkout")
        if not zone:
            st.checkout_start = None
        elif st.checkout_start is None:
            st.checkout_start = t
        elif t - st.checkout_start >= self.cfg.min_checkout_dwell:
            st.checked_out = True
            return [Event(t, st.track_id, Kind.CHECKOUT, zone.name, {})]
        return []

    def _exit(self, st: TrackState, p: Person, t: float) -> list[Event]:
        if st.exited:
            return []
        zone = self.zones.hit(p.feet, "exit")
        if zone:
            st.exited = True
            return [Event(t, st.track_id, Kind.EXIT, zone.name,
                          {"paid": st.checked_out, "picks": st.picks})]
        return []
