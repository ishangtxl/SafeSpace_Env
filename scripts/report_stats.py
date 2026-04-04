#!/usr/bin/env python3
"""Print benchmark statistics for the SafeSpace scenario corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server.scenarios import get_benchmark_manifest, get_scenario_statistics


def render_markdown(stats: dict) -> str:
    """Render a compact markdown summary for README updates."""
    lines = [
        "# SafeSpace Benchmark Stats",
        "",
        f"- Benchmark manifest version: {stats['benchmark_manifest_version']}",
        f"- Canonical benchmark size: {stats['canonical_total']}",
        f"- Total scenarios: {stats['total']}",
        f"- By difficulty: {stats['by_difficulty']}",
        f"- By decision: {stats['by_decision']}",
        f"- By trigger type: {stats['by_trigger_type']}",
        f"- Context depth overall: {stats['context_depth_overall']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report SafeSpace benchmark stats")
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format for the benchmark summary",
    )
    args = parser.parse_args()

    stats = get_scenario_statistics()
    manifest = get_benchmark_manifest()
    stats["benchmark_manifest_version"] = manifest["manifest_version"]
    stats["canonical_total"] = sum(len(ids) for ids in manifest["canonical"].values())
    if args.format == "markdown":
        print(render_markdown(stats))
    else:
        print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
