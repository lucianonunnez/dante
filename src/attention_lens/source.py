"""Where activation timelines come from.

Attention Lens consumes an :class:`ActivationTimeline`. In production that comes
from TRIBE's export (a ``.npz`` with per-second predicted fMRI + an aligned
transcript). Offline, it comes from :mod:`synth`. Both paths return the exact
same contract, so nothing downstream knows or cares which one ran — the seam is
here and only here.
"""
from __future__ import annotations

from pathlib import Path

from .schema import ActivationTimeline, Word
from .synth import generate as _generate


def from_mock(duration_s: int = 180, seed: int = 42) -> ActivationTimeline:
    return _generate(duration_s=duration_s, seed=seed)


def from_npz(path: str | Path) -> ActivationTimeline:
    """Load a TRIBE export.

    Expects arrays: ``fmri`` (seconds x voxels) and optional ``fmri_left`` /
    ``fmri_right``; plus an optional ``attention_json`` / ``words`` structure of
    ``[text, start, end]`` triples. Falls back gracefully if hemispheres or the
    transcript are absent.
    """
    import json

    import numpy as np

    data = np.load(path, allow_pickle=True)
    fmri = np.asarray(data["fmri"], dtype=float)
    activation = np.linalg.norm(fmri, axis=1).tolist()

    def norm(key: str) -> list[float]:
        return np.linalg.norm(np.asarray(data[key], dtype=float), axis=1).tolist() \
            if key in data else []

    left, right = norm("fmri_left"), norm("fmri_right")

    words: list[Word] = []
    for key in ("words", "attention_json"):
        if key in data:
            raw = data[key]
            try:
                items = json.loads(str(raw)) if raw.dtype.kind in "US" else list(raw)
            except Exception:
                items = list(raw)
            for it in items:
                try:
                    text, start, end = it[0], float(it[1]), float(it[2])
                    words.append(Word(text=str(text), start=start, end=end))
                except Exception:
                    continue
            break

    return ActivationTimeline(
        fps=1.0, activation=activation, left=left, right=right,
        words=words, source="tribe",
    )
