"""
ml/preprocessing/extract_keypoints.py
------------------------------------
Wrapper around legacy extract-keypoints-full.py so extraction can be invoked
from the new ml/preprocessing package path.

Usage:
    python -m ml.preprocessing.extract_keypoints
    python -m ml.preprocessing.extract_keypoints --run-as-main
"""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = PROJECT_ROOT / "extract-keypoints-full.py"


def run_via_import() -> object:
    """Load legacy script as a module and call extract_keypoints_with_splits()."""
    if not LEGACY_SCRIPT.exists():
        raise FileNotFoundError(f"Legacy script not found: {LEGACY_SCRIPT}")

    spec = importlib.util.spec_from_file_location("legacy_extract_keypoints", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {LEGACY_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["legacy_extract_keypoints"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "extract_keypoints_with_splits"):
        raise AttributeError("Legacy script missing extract_keypoints_with_splits()")

    print("[Preprocessing] Running legacy extraction function...")
    return module.extract_keypoints_with_splits()


def run_via_main() -> None:
    """Execute legacy script exactly as if run directly."""
    if not LEGACY_SCRIPT.exists():
        raise FileNotFoundError(f"Legacy script not found: {LEGACY_SCRIPT}")

    print("[Preprocessing] Running legacy script as __main__...")
    runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run keypoint extraction through the new preprocessing package")
    parser.add_argument(
        "--run-as-main",
        action="store_true",
        help="Run extract-keypoints-full.py as __main__ (includes its post-analysis prints)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.run_as_main:
        run_via_main()
    else:
        run_via_import()


if __name__ == "__main__":
    main()
