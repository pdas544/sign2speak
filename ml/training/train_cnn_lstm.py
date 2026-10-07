"""
ml/training/train_cnn_lstm.py
-----------------------------
Entry point for CNN+LSTM model training.

Supports two dataset modes:
1) processed: loads train/val/test arrays produced by ml.preprocessing.build_dataset
2) legacy: uses ActionRecognitionModel.load_data over mp_data/test_data style folders

Writes structured per-run outputs under:

    outputs/training/cnn_lstm/<run_id>/
        checkpoints/   <- .h5 checkpoints
        plots/         <- training history, confusion matrix images
        reports/       <- classification report .txt

Also writes a manifest and registers the model in models/registry/registry.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import to_categorical

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from action_recognition_model import ActionRecognitionModel
from test_action_recognition import SignLanguageDetector
from ml.training.common import create_run_dir, register_model, write_manifest


DEFAULT_MODEL_NAME = "cnn_lstm_default"
MODEL_FAMILY = "cnn_lstm"
SEQUENCE_LENGTH = 30
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "processed" / "datasets" / "cnn_lstm"


def _load_processed_dataset(dataset_root: Path):
    required = [
        "train_X.npy",
        "train_y.npy",
        "val_X.npy",
        "val_y.npy",
        "test_X.npy",
        "test_y.npy",
        "labels.json",
    ]
    missing = [name for name in required if not (dataset_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Processed dataset missing files under {dataset_root}: {missing}. "
            "Run: python -m ml.preprocessing.build_dataset"
        )

    labels_payload = json.loads((dataset_root / "labels.json").read_text(encoding="utf-8"))
    actions = list(labels_payload.get("labels", []))
    if not actions:
        raise ValueError(f"No labels found in {dataset_root / 'labels.json'}")

    X_train = np.load(dataset_root / "train_X.npy").astype(np.float32)
    y_train_idx = np.load(dataset_root / "train_y.npy").astype(np.int64)
    X_val = np.load(dataset_root / "val_X.npy").astype(np.float32)
    y_val_idx = np.load(dataset_root / "val_y.npy").astype(np.int64)
    X_test = np.load(dataset_root / "test_X.npy").astype(np.float32)
    y_test_idx = np.load(dataset_root / "test_y.npy").astype(np.int64)

    num_classes = len(actions)
    y_train = to_categorical(y_train_idx, num_classes=num_classes).astype(np.int32)
    y_val = to_categorical(y_val_idx, num_classes=num_classes).astype(np.int32)
    y_test = to_categorical(y_test_idx, num_classes=num_classes).astype(np.int32)

    return X_train, y_train, X_val, y_val, X_test, y_test, actions


def _load_legacy_dataset(detector: SignLanguageDetector, arm: ActionRecognitionModel):
    data_path = detector.data_path
    print(f"[Training] Legacy dataset path: {data_path}")
    X_train, X_test, y_train, y_test = arm.load_data(data_path)
    return X_train, X_test, y_train, y_test


def train(
    model_name: str = DEFAULT_MODEL_NAME,
    set_active: bool = False,
    dataset_mode: str = "auto",
    dataset_root: Path = DEFAULT_PROCESSED_ROOT,
) -> Path:
    """
    Run a full CNN+LSTM training cycle and save outputs to a structured run directory.

    dataset_mode:
      - auto: use processed dataset when available, else fallback to legacy
      - processed: require processed arrays from ml.preprocessing.build_dataset
      - legacy: use existing folder-based loader in ActionRecognitionModel
    """
    run_dir = create_run_dir(MODEL_FAMILY)
    checkpoint_path = str(run_dir / "checkpoints" / f"{model_name}.h5")

    detector = SignLanguageDetector()
    actions = list(detector.actions)

    arm = ActionRecognitionModel(
        actions=actions,
        sequence_length=SEQUENCE_LENGTH,
        model_path=checkpoint_path,
    )

    reports_dir = str(run_dir / "reports")
    plots_dir = str(run_dir / "plots")
    arm.logs_dir = reports_dir
    os.makedirs(reports_dir, exist_ok=True)

    resolved_mode = dataset_mode
    processed_available = (dataset_root / "labels.json").exists()
    if dataset_mode == "auto":
        resolved_mode = "processed" if processed_available else "legacy"

    if resolved_mode == "processed":
        print(f"\n[Training] Loading processed dataset from: {dataset_root}")
        X_train, y_train, X_val, y_val, X_test, y_test, actions = _load_processed_dataset(dataset_root)
        arm.actions = actions
        arm.label_map = {str(label): idx for idx, label in enumerate(actions)}
        X_eval, y_eval = X_test, y_test
    elif resolved_mode == "legacy":
        print("\n[Training] Loading legacy dataset via ActionRecognitionModel.load_data(...)")
        X_train, X_test, y_train, y_test = _load_legacy_dataset(detector, arm)
        X_val, y_val = X_test, y_test
        X_eval, y_eval = X_test, y_test
    else:
        raise ValueError("dataset_mode must be one of: auto, processed, legacy")

    input_features = int(X_train.shape[2])

    print(f"[Training] Dataset mode: {resolved_mode}")
    print(f"[Training] Classes ({len(actions)}): {actions}")
    print(f"[Training] Input features: {input_features}")
    print(f"[Training] Train shape: {X_train.shape} | Val shape: {X_val.shape}")
    print(f"[Training] Checkpoint: {checkpoint_path}")

    print("\n[Training] Starting training...")
    arm.train_model(X_train, y_train, X_val, y_val)

    for suffix in (".png", ".npy"):
        for f in Path(reports_dir).glob(f"*{suffix}"):
            dest = Path(plots_dir) / f.name
            shutil.copy2(str(f), str(dest))

    print("\n[Training] Evaluating...")
    arm.logs_dir = reports_dir
    results = arm.evaluate_model(X_eval, y_eval)
    print(f"[Training] Accuracy: {results['accuracy']:.4f}")

    write_manifest(
        run_dir,
        model_name=model_name,
        framework="tf",
        model_path=checkpoint_path,
        labels=actions,
        sequence_length=SEQUENCE_LENGTH,
        input_features=input_features,
        description="CNN+LSTM trained via unified ml.preprocessing/ml.training pipeline",
        extra={
            "accuracy": float(results["accuracy"]),
            "dataset_mode": resolved_mode,
            "dataset_root": str(dataset_root),
        },
    )

    register_model(
        model_name,
        display_name=f"CNN-LSTM ({model_name})",
        framework="tf",
        model_path=checkpoint_path,
        labels=actions,
        sequence_length=SEQUENCE_LENGTH,
        input_features=input_features,
        description="CNN+LSTM trained via unified ml.preprocessing/ml.training pipeline",
        set_active=set_active,
    )

    print(f"\n[Training] Done. Model saved to: {checkpoint_path}")
    return Path(checkpoint_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN+LSTM sign-language model")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Registry key for the trained model (default: %(default)s)",
    )
    parser.add_argument(
        "--set-active",
        action="store_true",
        help="Mark this model as the active inference model in the registry",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["auto", "processed", "legacy"],
        default="auto",
        help="Dataset source: auto-detect processed else legacy (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_PROCESSED_ROOT),
        help="Directory containing train_X/train_y/val_X/val_y/test_X/test_y/labels.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(
        model_name=args.model_name,
        set_active=args.set_active,
        dataset_mode=args.dataset_mode,
        dataset_root=Path(args.dataset_root),
    )


if __name__ == "__main__":
    main()
