"""Transparent risk score (0-100). Every point can be traced back to a reason."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import TrackState

LOW, MEDIUM, HIGH = "LOW", "MEDIUM", "HIGH"


@dataclass
class RiskResult:
    score: int
    level: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": self.score, "level": self.level, "reasons": self.reasons}


def score_track(st: TrackState) -> RiskResult:
    score, reasons = 0, []
    if st.picks:
        pts = min(st.picks, 5) * 6
        score += pts
        reasons.append(f"+{pts}: {st.picks} item pick(s) from shelves")
    if st.conceals:
        pts = min(st.conceals, 3) * 25
        score += pts
        reasons.append(f"+{pts}: {st.conceals} concealment gesture(s) after a pick")
    if st.picks and st.exited and not st.checked_out:
        score += 35
        reasons.append("+35: left the store with picked items and no checkout event")
    if st.checked_out:
        pts = 30 if not st.conceals else 10
        score -= pts
        reasons.append(f"-{pts}: spent time at the checkout")
    score = max(0, min(100, score))
    level = LOW if score < 25 else MEDIUM if score < 55 else HIGH
    return RiskResult(score, level, reasons)
