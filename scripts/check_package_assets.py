#!/usr/bin/env python3
"""Build a wheel from a clean source copy and verify packaged runtime assets."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
EXPECTED_ASSETS = [
    Path("content_moderation_env/openenv.yaml"),
    Path("content_moderation_env/server/data/benchmark_manifest.json"),
    Path("content_moderation_env/server/data/scenarios_easy.json"),
    Path("content_moderation_env/server/data/scenarios_medium.json"),
    Path("content_moderation_env/server/data/scenarios_hard.json"),
]


def _copy_source_tree(destination: Path) -> Path:
    """Copy the project into a temporary clean tree for packaging checks."""
    source_copy = destination / "source"
    ignore = shutil.ignore_patterns(
        ".git",
        ".venv",
        ".review-venv",
        "__pycache__",
        ".pytest_cache",
        ".coverage",
        "build",
        "dist",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "openenv_safespace.egg-info",
    )
    shutil.copytree(ROOT_DIR, source_copy, ignore=ignore)
    return source_copy


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return captured output or raise on failure."""
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _verify_installed_import(install_dir: Path) -> int:
    """Import the installed wheel and validate the scenario corpus from disk."""
    command = [
        sys.executable,
        "-c",
        (
            "import sys;"
            "sys.path.insert(0, sys.argv[1]);"
            "from content_moderation_env.server.scenarios import validate_scenario_corpus;"
            "corpus = validate_scenario_corpus();"
            "print(sum(len(items) for items in corpus.values()))"
        ),
        str(install_dir),
    ]
    result = _run_command(command)
    return int(result.stdout.strip())


def main() -> None:
    """Build a wheel, install it into a temp target, and verify asset presence."""
    with tempfile.TemporaryDirectory(prefix="safespace-package-") as tmpdir:
        temp_root = Path(tmpdir)
        source_copy = _copy_source_tree(temp_root)
        wheel_dir = temp_root / "wheelhouse"
        install_dir = temp_root / "install"
        wheel_dir.mkdir()
        install_dir.mkdir()

        _run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(source_copy),
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
            ]
        )

        wheels = sorted(wheel_dir.glob("openenv_safespace-*.whl"))
        if not wheels:
            raise SystemExit("No wheel was produced by the packaging smoke check.")
        wheel_path = wheels[0]

        _run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(install_dir),
                str(wheel_path),
            ]
        )

        missing_assets = [
            str(asset)
            for asset in EXPECTED_ASSETS
            if not (install_dir / asset).exists()
        ]
        corpus_total = _verify_installed_import(install_dir)

        payload = {
            "status": "ok" if not missing_assets else "missing_assets",
            "wheel": wheel_path.name,
            "installed_corpus_total": corpus_total,
            "expected_assets": [str(asset) for asset in EXPECTED_ASSETS],
            "missing_assets": missing_assets,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

        if missing_assets:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
