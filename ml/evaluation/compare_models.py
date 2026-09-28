"""
ml/evaluation/compare_models.py
--------------------------------
Compare multiple registered models on the same dataset split.

Usage:
    python -m ml.evaluation.compare_models --models cnn_lstm_default transformer_v1
    python -m ml.evaluation.compare_models --all-models --split test --max-samples 300
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_registry_service import ModelRegistryService
from ml.evaluation.evaluate_model import evaluate_model


DEFAULT_METADATA = PROJECT_ROOT / "processed" / "metadata.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluation"


def compare_models(
    model_names: list[str],
    metadata_path: Path = DEFAULT_METADATA,
    split: str = "test",
    max_samples: int | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for model_name in model_names:
        print(f"\n[Compare] Evaluating model: {model_name}")
        summary = evaluate_model(
            model_name=model_name,
            metadata_path=metadata_path,
            split=split,
            max_samples=max_samples,
            output_root=output_root,
        )
        results.append(summary)

    results_sorted = sorted(results, key=lambda row: row.get("accuracy", 0.0), reverse=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    compare_dir = output_root / "comparisons" / run_id
    compare_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": split,
        "metadata_path": str(metadata_path),
        "max_samples": max_samples,
        "compared_models": model_names,
        "leaderboard": results_sorted,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    (compare_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with (compare_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_name",
                "framework",
                "split",
                "samples_total",
                "samples_successful",
                "samples_skipped",
                "accuracy",
                "f1_macro",
                "predict_ms_p50",
                "predict_ms_p95",
                "artifact_mb",
                "output_dir",
            ],
        )
        writer.writeheader()
        for row in results_sorted:
            writer.writerow(
                {
                    "model_name": row.get("model_name"),
                    "framework": row.get("framework"),
                    "split": row.get("split"),
                    "samples_total": row.get("samples_total"),
                    "samples_successful": row.get("samples_successful"),
                    "samples_skipped": row.get("samples_skipped"),
                    "accuracy": row.get("accuracy"),
                    "f1_macro": row.get("f1_macro"),
                    "predict_ms_p50": row.get("predict_ms_p50"),
                    "predict_ms_p95": row.get("predict_ms_p95"),
                    "artifact_mb": row.get("artifact_mb"),
                    "output_dir": row.get("output_dir"),
                }
            )

    print("\n[Compare] Leaderboard")
    for idx, row in enumerate(results_sorted, start=1):
        print(
            f"{idx:>2}. {row['model_name']:<24} "
            f"acc={row['accuracy']:.4f} "
            f"f1_macro={row['f1_macro']:.4f} "
            f"p50={row.get('predict_ms_p50', float('nan')):.1f}ms "
            f"p95={row.get('predict_ms_p95', float('nan')):.1f}ms "
            f"size={row.get('artifact_mb', float('nan'))}MB "
            f"(n={row['samples_successful']})"
        )

    print("\n[Compare] Summary JSON:", compare_dir / "comparison_summary.json")
    print("[Compare] Leaderboard CSV:", compare_dir / "leaderboard.csv")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare registered models on the same split")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit list of registry model keys",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Compare all models from the registry",
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to metadata CSV")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per model")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for artifacts")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    registry = ModelRegistryService()
    registry_models = [m["name"] for m in registry.list_models()]

    if args.all_models:
        model_names = registry_models
    else:
        model_names = args.models or []

    if not model_names:
        raise SystemExit("No models selected. Use --models ... or --all-models")

    unknown = [name for name in model_names if name not in registry_models]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Known models: {registry_models}")

    compare_models(
        model_names=model_names,
        metadata_path=Path(args.metadata),
        split=args.split,
        max_samples=args.max_samples,
        output_root=Path(args.output_root),
    )


if __name__ == "__main__":
    main()
