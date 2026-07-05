"""Synthetic activation generator.

The real signal comes from TRIBE (GPU + model weights). So the package can run,
be tested and be demoed anywhere, this module fabricates a plausible per-second
activation timeline plus a matching transcript, with clearly-placed hooks and
dead-air stretches. Seeded, so it's reproducible.
"""
from __future__ import annotations

import math
import random

from .schema import ActivationTimeline, Word

_WORDS = (
    "and so the thing about this is that when you look closely you start to see "
    "a pattern nobody really talks about which changes how the whole story lands "
    "for the people watching at home right now today"
).split()


def generate(duration_s: int = 180, seed: int = 42) -> ActivationTimeline:
    """Fabricate an activation timeline with a couple of hooks and dead spots."""
    rng = random.Random(seed)

    # Baseline slow drift + noise.
    base = [0.5 + 0.15 * math.sin(t / 20.0) + rng.gauss(0, 0.05) for t in range(duration_s)]

    # Plant two hooks (sustained high) and two dead-air stretches (sustained low).
    def plateau(center: int, half: int, level: float):
        for t in range(max(0, center - half), min(duration_s, center + half)):
            base[t] += level

    plateau(int(duration_s * 0.15), 6, +0.6)   # early hook
    plateau(int(duration_s * 0.70), 8, +0.7)   # strong later hook
    plateau(int(duration_s * 0.40), 10, -0.4)  # mid dead air
    plateau(int(duration_s * 0.88), 7, -0.45)  # late dead air

    activation = [max(0.01, v) for v in base]
    left = [max(0.01, v * rng.uniform(0.9, 1.1)) for v in activation]
    right = [max(0.01, v * rng.uniform(0.9, 1.1)) for v in activation]

    # A transcript at ~2.5 words/second.
    words: list[Word] = []
    t = 0.0
    i = 0
    while t < duration_s:
        w = _WORDS[i % len(_WORDS)]
        dur = rng.uniform(0.3, 0.6)
        words.append(Word(text=w, start=round(t, 2), end=round(t + dur, 2)))
        t += dur
        i += 1

    return ActivationTimeline(
        fps=1.0, activation=activation, left=left, right=right,
        words=words, source="mock",
    )
