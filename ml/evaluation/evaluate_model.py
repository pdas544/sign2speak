"""
ml/evaluation/evaluate_model.py
--------------------------------
Evaluate one registered model against a labeled split from metadata.

Default input format:
- metadata CSV with columns: file_path,label,split,status
- feature files referenced by file_path can be .pt or .npy

Usage:
    python -m ml.evaluation.evaluate_model --model-name cnn_lstm_default
    python -m ml.evaluation.evaluate_model --model-name transformer_v1 --split test
    python -m ml.evaluation.evaluate_model --model-name transformer_v1 --max-samples 300
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_registry_service import ModelRegistryService


DEFAULT_METADATA = PROJECT_ROOT / "processed" / "metadata.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluation"


@dataclass(frozen=True)
class EvalSample:
    file_path: Path
    label: str
    split: str


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _load_keypoints(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".pt":
        import torch

        tensor = torch.load(path, map_location="cpu")
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
    else:
        raise ValueError(f"Unsupported feature file extension: {path.suffix}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D keypoint array, got shape {arr.shape}")

    return arr


def _normalize_for_model(
    seq: np.ndarray,
    sequence_length: int,
    input_features: int,
) -> np.ndarray:
    # Align feature dimension first.
    if seq.shape[1] > input_features:
        seq = seq[:, :input_features]
    elif seq.shape[1] < input_features:
        pad_cols = input_features - seq.shape[1]
        seq = np.pad(seq, ((0, 0), (0, pad_cols)), mode="constant")

    # Align temporal length by keeping the most recent frames.
    if seq.shape[0] >= sequence_length:
        seq = seq[-sequence_length:]
    else:
        pad_rows = sequence_length - seq.shape[0]
        seq = np.pad(seq, ((pad_rows, 0), (0, 0)), mode="constant")

    return seq.astype(np.float32)


def _iter_metadata_samples(
    metadata_path: Path,
    split: str,
    allowed_labels: set[str],
) -> list[EvalSample]:
    samples: list[EvalSample] = []

    with metadata_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_status = (row.get("status") or "").lower()
            row_split = (row.get("split") or "").strip()
            row_label = (row.get("label") or "").strip()
            row_path = (row.get("file_path") or "").strip()

            if "success" not in row_status:
                continue
            if row_split != split:
                continue
            if row_label not in allowed_labels:
                continue
            if not row_path:
                continue

            samples.append(EvalSample(file_path=PROJECT_ROOT / row_path, label=row_label, split=row_split))

    return samples


def _save_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        fig_w = max(8, int(len(labels) * 0.8))
        fig_h = max(6, int(len(labels) * 0.6))

        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar(fraction=0.046, pad=0.04)

        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=90)
        plt.yticks(ticks, labels)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
    except Exception as exc:
        print(f"[Evaluation] Warning: could not save confusion matrix image: {exc}")


def evaluate_model(
    model_name: str,
    metadata_path: Path = DEFAULT_METADATA,
    split: str = "test",
    max_samples: int | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    registry = ModelRegistryService()
    model_config = registry.get_model_config(model_name)
    backend = registry.build_backend(model_config)

    labels: list[str] = list(model_config.get("labels", []))
    sequence_length = int(model_config.get("sequence_length", 30))
    input_features = int(model_config.get("input_features", 1662))

    if not labels:
        raise ValueError(f"No labels configured for model '{model_name}'")

    samples = _iter_metadata_samples(metadata_path, split, set(labels))
    if max_samples is not None:
        samples = samples[:max_samples]

    if not samples:
        raise RuntimeError(
            f"No evaluation samples found for split='{split}' with labels in model '{model_name}'"
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    skipped = 0
    for idx, sample in enumerate(samples, start=1):
        try:
            import time as _time

            seq = _load_keypoints(sample.file_path)
            seq = _normalize_for_model(seq, sequence_length=sequence_length, input_features=input_features)
            _t0 = _time.perf_counter()
            pred_label, confidence = backend.predict(seq)
            latencies_ms.append((_time.perf_counter() - _t0) * 1000.0)

            y_true.append(sample.label)
            y_pred.append(pred_label)
            rows.append(
                {
                    "index": idx,
                    "file_path": str(sample.file_path),
                    "label_true": sample.label,
                    "label_pred": pred_label,
                    "confidence": confidence,
                    "correct": int(sample.label == pred_label),
                }
            )
        except Exception as exc:
            skipped += 1
            rows.append(
                {
                    "index": idx,
                    "file_path": str(sample.file_path),
                    "label_true": sample.label,
                    "label_pred": "<error>",
                    "confidence": math.nan,
                    "correct": 0,
                    "error": str(exc),
                }
            )

    if not y_true:
        raise RuntimeError("All samples failed during evaluation")

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "file_path", "label_true", "label_pred", "confidence", "correct", "error"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row.get("index"),
                    "file_path": row.get("file_path"),
                    "label_true": row.get("label_true"),
                    "label_pred": row.get("label_pred"),
                    "confidence": _safe_float(row.get("confidence")),
                    "correct": row.get("correct"),
                    "error": row.get("error", ""),
                }
            )

    _save_confusion_matrix(cm, labels, run_dir / "confusion_matrix.png")

    lat = np.asarray(latencies_ms, dtype=np.float64)
    predict_ms_p50 = float(np.median(lat)) if lat.size else math.nan
    predict_ms_p95 = float(np.percentile(lat, 95)) if lat.size else math.nan
    try:
        artifact_mb = round(
            (PROJECT_ROOT / str(model_config.get("model_path", ""))).stat().st_size
            / (1024 * 1024), 3,
        )
    except OSError:
        artifact_mb = math.nan

    summary = {
        "model_name": model_name,
        "framework": model_config.get("framework"),
        "split": split,
        "metadata_path": str(metadata_path),
        "samples_total": len(samples),
        "samples_successful": len(y_true),
        "samples_skipped": skipped,
        "sequence_length": sequence_length,
        "input_features": input_features,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "predict_ms_p50": predict_ms_p50,
        "predict_ms_p95": predict_ms_p95,
        "artifact_mb": artifact_mb,
        "labels_count": len(labels),
        "output_dir": str(run_dir),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write back headline metrics onto the registry entry (best-effort).
    try:
        from ml.training.common import update_model_metrics

        update_model_metrics(model_name, {
            "test_accuracy": acc,
            "test_f1_macro": f1_macro,
            "predict_ms_p50": predict_ms_p50,
            "predict_ms_p95": predict_ms_p95,
            "artifact_mb": artifact_mb,
            "eval_run_id": run_dir.name,
            "evaluated_at": summary["evaluated_at"],
        })
    except Exception as exc:
        print(f"[Evaluation] Warning: registry write-back failed: {exc}")

    print("\n[Evaluation] Completed")
    print(json.dumps(summary, indent=2))
    print("\n[Evaluation] report:", run_dir / "classification_report.txt")
    print("[Evaluation] predictions:", run_dir / "predictions.csv")
    print("[Evaluation] confusion matrix:", run_dir / "confusion_matrix.png")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one registered model")
    parser.add_argument("--model-name", required=True, help="Registry key of model to evaluate")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA), help="Path to metadata CSV")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick evaluation")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for eval artifacts")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    evaluate_model(
        model_name=args.model_name,
        metadata_path=Path(args.metadata),
        split=args.split,
        max_samples=args.max_samples,
        output_root=Path(args.output_root),
    )


if __name__ == "__main__":
    main()
