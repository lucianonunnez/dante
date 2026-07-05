"""Typed contracts shared across Attention Lens.

The unit of analysis is a **video second**. TRIBE predicts a vector of cortical
activation for every second of a video; Attention Lens turns that stream into an
engagement signal and, from there, into decisions ("this stretch is dead air").
Keeping the data shapes here means the model seam, the metrics and the report
never disagree on what a "timeline" is.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    """A transcript token with its time span, in seconds."""
    text: str
    start: float
    end: float


@dataclass
class ActivationTimeline:
    """Raw per-second cortical activation coming out of the model (or the mock).

    ``activation`` is one scalar per second (the L2 norm of the predicted fMRI
    vector). ``left`` / ``right`` optionally carry the per-hemisphere norms.
    """
    fps: float                       # video seconds per sample (usually 1.0)
    activation: list[float]
    left: list[float] = field(default_factory=list)
    right: list[float] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    source: str = "unknown"          # "tribe" | "mock" | ...

    @property
    def duration_s(self) -> float:
        return len(self.activation) / self.fps


@dataclass
class Segment:
    """A contiguous stretch of the video with a shared engagement character."""
    kind: str                        # "hook" | "dead_air"
    start_s: float
    end_s: float
    mean_engagement: float           # 0..100
    quote: str = ""                  # transcript words spoken in the window

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class EngagementReport:
    duration_s: float
    overall_score: float             # 0..100, area under the engagement curve
    engagement: list[float]          # per-second, 0..100
    hooks: list[Segment] = field(default_factory=list)
    dead_air: list[Segment] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    source: str = "unknown"
