"""Render an :class:`EngagementReport` as Markdown and a compact ASCII sparkline."""
from __future__ import annotations

from .schema import EngagementReport

_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return ""
    # Down-sample to `width` buckets.
    step = max(1, len(values) // width)
    buckets = [values[i:i + step] for i in range(0, len(values), step)]
    means = [sum(b) / len(b) for b in buckets]
    lo, hi = min(means), max(means)
    span = (hi - lo) or 1.0
    return "".join(_SPARK[min(len(_SPARK) - 1, int((m - lo) / span * (len(_SPARK) - 1)))]
                    for m in means)


def _fmt(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    return f"{m:d}:{s:02d}"


def to_markdown(r: EngagementReport) -> str:
    lines = [
        "# Engagement report",
        "",
        f"- **Duration:** {_fmt(r.duration_s)}",
        f"- **Overall engagement:** {r.overall_score:.0f}/100",
        f"- **Source:** {r.source}",
        "",
        "```",
        sparkline(r.engagement),
        "```",
        "",
        "## Hooks (lead with these)",
    ]
    if r.hooks:
        for h in r.hooks[:5]:
            lines.append(f"- **{_fmt(h.start_s)}–{_fmt(h.end_s)}** "
                         f"(engagement {h.mean_engagement:.0f}): \"{h.quote}\"")
    else:
        lines.append("- none detected")

    lines += ["", "## Dead air (candidates to trim)"]
    if r.dead_air:
        for d in r.dead_air[:5]:
            lines.append(f"- **{_fmt(d.start_s)}–{_fmt(d.end_s)}** "
                         f"({d.duration_s:.0f}s, engagement {d.mean_engagement:.0f})")
    else:
        lines.append("- none detected")

    lines += ["", "## Recommendations"]
    lines += [f"- {rec}" for rec in r.recommendations]
    return "\n".join(lines)
