"""Turn a raw activation timeline into an engagement signal.

The pipeline is deliberately simple and explainable — an editor has to trust
why a second scored low:

1. **Smooth** the raw per-second activation (rolling mean) to remove jitter.
2. **Normalise** to a 0..100 engagement index (robust min/max on percentiles).
3. **Segment** into *hooks* (sustained high engagement) and *dead air*
   (sustained low engagement).

No learned post-processing: the model already did the hard part. This layer is
transparent arithmetic on top of it.
"""
from __future__ import annotations

from .schema import ActivationTimeline, EngagementReport, Segment, Word


def _rolling_mean(xs: list[float], window: int) -> list[float]:
    if window <= 1 or len(xs) < 2:
        return list(xs)
    out = []
    half = window // 2
    for i in range(len(xs)):
        lo, hi = max(0, i - half), min(len(xs), i + half + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def _percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    idx = min(len(sorted_xs) - 1, max(0, int(round(q * (len(sorted_xs) - 1)))))
    return sorted_xs[idx]


def _normalise(xs: list[float]) -> list[float]:
    """Map to 0..100 using the 5th/95th percentiles as the robust range so a
    single spike doesn't flatten everything else."""
    s = sorted(xs)
    lo, hi = _percentile(s, 0.05), _percentile(s, 0.95)
    if hi - lo < 1e-9:
        return [50.0 for _ in xs]
    out = []
    for x in xs:
        v = (x - lo) / (hi - lo) * 100
        out.append(float(min(100.0, max(0.0, v))))
    return out


def _quote(words: list[Word], start_s: float, end_s: float, max_words: int = 25) -> str:
    spoken = [w.text for w in words if w.start < end_s and w.end > start_s]
    text = " ".join(spoken)
    parts = text.split()
    if len(parts) > max_words:
        text = " ".join(parts[:max_words]) + " ..."
    return text


def _segments(engagement: list[float], fps: float, words: list[Word],
              kind: str, predicate, min_len_s: float) -> list[Segment]:
    segs: list[Segment] = []
    run_start = None
    for i, e in enumerate(engagement + [None]):  # sentinel to close last run
        if e is not None and predicate(e):
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                start_s, end_s = run_start / fps, i / fps
                if end_s - start_s >= min_len_s:
                    window = engagement[run_start:i]
                    segs.append(Segment(
                        kind=kind, start_s=start_s, end_s=end_s,
                        mean_engagement=sum(window) / len(window),
                        quote=_quote(words, start_s, end_s),
                    ))
                run_start = None
    return segs


def analyse(
    timeline: ActivationTimeline,
    smooth_window: int = 3,
    hook_threshold: float = 70.0,
    dead_threshold: float = 30.0,
    min_segment_s: float = 3.0,
) -> EngagementReport:
    """Compute the engagement curve and extract hooks / dead-air segments."""
    smoothed = _rolling_mean(timeline.activation, smooth_window)
    engagement = _normalise(smoothed)
    fps = timeline.fps

    hooks = _segments(engagement, fps, timeline.words, "hook",
                      lambda e: e >= hook_threshold, min_segment_s)
    dead = _segments(engagement, fps, timeline.words, "dead_air",
                     lambda e: e <= dead_threshold, min_segment_s)

    overall = sum(engagement) / len(engagement) if engagement else 0.0

    report = EngagementReport(
        duration_s=timeline.duration_s,
        overall_score=round(overall, 1),
        engagement=[round(e, 1) for e in engagement],
        hooks=sorted(hooks, key=lambda s: s.mean_engagement, reverse=True),
        dead_air=sorted(dead, key=lambda s: s.duration_s, reverse=True),
        source=timeline.source,
    )
    report.recommendations = _recommend(report)
    return report


def _fmt(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    return f"{m:d}:{s:02d}"


def _recommend(r: EngagementReport) -> list[str]:
    recs: list[str] = []
    if r.overall_score >= 65:
        recs.append(f"Strong overall engagement ({r.overall_score:.0f}/100). Keep the pacing.")
    elif r.overall_score >= 45:
        recs.append(f"Moderate engagement ({r.overall_score:.0f}/100). Tighten the slow stretches below.")
    else:
        recs.append(f"Low overall engagement ({r.overall_score:.0f}/100). Consider a shorter cut.")

    for d in r.dead_air[:3]:
        recs.append(
            f"Trim {_fmt(d.start_s)}–{_fmt(d.end_s)} ({d.duration_s:.0f}s, "
            f"engagement {d.mean_engagement:.0f}) — dead air."
        )
    if r.hooks:
        h = r.hooks[0]
        recs.append(
            f"Lead with {_fmt(h.start_s)}–{_fmt(h.end_s)} (engagement "
            f"{h.mean_engagement:.0f}) — your strongest hook."
        )
    return recs
