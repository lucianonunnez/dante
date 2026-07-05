"""Offline, deterministic tests for the engagement pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_lens import analyse, source, to_markdown
from attention_lens.engagement import _normalise


def test_normalise_range():
    out = _normalise([0, 1, 2, 3, 4, 5, 100])
    assert min(out) >= 0 and max(out) <= 100


def test_mock_timeline_shape():
    tl = source.from_mock(duration_s=120, seed=1)
    assert tl.duration_s == 120
    assert len(tl.activation) == 120
    assert tl.words and tl.source == "mock"


def test_analyse_detects_hooks_and_dead_air():
    tl = source.from_mock(duration_s=180, seed=42)
    r = analyse(tl)
    assert 0 <= r.overall_score <= 100
    assert len(r.engagement) == 180
    # The synthetic clip plants both, so the analyser must find both.
    assert r.hooks, "expected at least one hook"
    assert r.dead_air, "expected at least one dead-air stretch"
    assert r.recommendations


def test_hooks_have_quotes_from_transcript():
    r = analyse(source.from_mock(seed=7))
    assert any(h.quote for h in r.hooks)


def test_report_renders_markdown():
    md = to_markdown(analyse(source.from_mock(seed=3)))
    assert "# Engagement report" in md
    assert "Recommendations" in md


def test_deterministic():
    a = analyse(source.from_mock(seed=5)).overall_score
    b = analyse(source.from_mock(seed=5)).overall_score
    assert a == b
