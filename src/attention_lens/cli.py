"""Command-line interface.

    attention-lens demo                 # analyse a synthetic clip, print the report
    attention-lens analyze run.npz      # analyse a real TRIBE export
    attention-lens analyze run.npz --json report.json

Kept dependency-light (argparse only) so it runs without a framework.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from . import source
from .engagement import analyse
from .report import to_markdown


def _run(timeline, as_json: str | None) -> int:
    report = analyse(timeline)
    if as_json:
        with open(as_json, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Wrote {as_json}")
    else:
        print(to_markdown(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="attention-lens",
                                     description="Video engagement analytics from cortical activation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="analyse a synthetic clip (no model needed)")
    p_demo.add_argument("--duration", type=int, default=180)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--json", dest="as_json", default=None)

    p_an = sub.add_parser("analyze", help="analyse a TRIBE .npz export")
    p_an.add_argument("npz")
    p_an.add_argument("--json", dest="as_json", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return _run(source.from_mock(args.duration, args.seed), args.as_json)
    if args.cmd == "analyze":
        return _run(source.from_npz(args.npz), args.as_json)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
