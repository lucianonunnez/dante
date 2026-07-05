"""Attention Lens — video engagement analytics from cortical activation.

Public surface:
    from attention_lens import analyse, source, to_markdown
"""
from .engagement import analyse
from .report import to_markdown, sparkline
from .schema import ActivationTimeline, EngagementReport, Segment, Word
from . import source

__all__ = [
    "analyse", "to_markdown", "sparkline", "source",
    "ActivationTimeline", "EngagementReport", "Segment", "Word",
]
