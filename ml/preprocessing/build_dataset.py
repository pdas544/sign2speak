"""
ml/preprocessing/build_dataset.py
---------------------------------
Build train/val/test numpy datasets from mp_data/<label>/<sequence>/<frame>.npy.

This script enforces split-before-augmentation, which prevents train/test leakage.

Outputs:
  <output_root>/
    labels.json
    metadata.csv
    train_X.npy, train_y.npy
    val_X.npy, val_y.npy
    test_X.npy, test_y.npy

Usage:
  python -m ml.preprocessing.build_dataset
  python -m ml.preprocessing.build_dataset --augment-copies 1 --sequence-length 30
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from ml.preprocessing.augment import augment_sequence, pad_or_truncate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "mp_data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "processed" / "datasets" / "cnn_lstm"


@dataclass(frozen=True)
class SequenceSample:
    sequence_path: Path
    label: str


def _numeric_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), stem) if stem.isdigit() else (10**9, stem)


def _scan_sequences(input_root: Path) -> list[SequenceSample]:
    samples: list[SequenceSample] = []

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    for label_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for seq_dir in sorted([p for p in label_dir.iterdir() if p.is_dir()], key=lambda p: _numeric_key(p)):
            frame_files = sorted(seq_dir.glob("*.npy"), key=lambda p: _numeric_key(p))
            if not frame_files:
                continue
            samples.append(SequenceSample(sequence_path=seq_dir, label=label))

    if not samples:
        raise RuntimeError(f"No sequence directories with .npy frames found under: {input_root}")

    return samples


def _load_sequence(sequence_dir: Path, sequence_length: int) -> np.ndarray:
    frame_files = sorted(sequence_dir.glob("*.npy"), key=lambda p: _numeric_key(p))
    frames = [np.load(str(fp)).astype(np.float32) for fp in frame_files]

    seq = np.asarray(frames, dtype=np.float32)
    if seq.ndim != 2:
        # Expected shape: (frames, features)
        seq = np.reshape(seq, (seq.shape[0], -1)).astype(np.float32)

    return pad_or_truncate(seq, target_length=sequence_length)


def _to_arrays(samples: list[SequenceSample], sequence_length: int, label_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[int] = []

    for sample in samples:
        seq = _load_sequence(sample.sequence_path, sequence_length=sequence_length)
        X.append(seq)
        y.append(label_to_idx[sample.label])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _augment_split(
    X: np.ndarray,
    y: np.ndarray,
    copies: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if copies <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    augmented: list[np.ndarray] = [X]
    labels: list[np.ndarray] = [y]

    for _ in range(copies):
        X_aug = np.asarray([augment_sequence(seq, rng=rng) for seq in X], dtype=np.float32)
        augmented.append(X_aug)
        labels.append(y.copy())

    return np.concatenate(augmented, axis=0), np.concatenate(labels, axis=0)


def build_dataset(
    input_root: Path,
    output_root: Path,
    sequence_length: int = 30,
    test_size: float = 0.2,
    val_size: float = 0.1,
    augment_copies: int = 0,
    seed: int = 42,
) -> dict[str, object]:
    samples = _scan_sequences(input_root)
    labels = sorted({s.label for s in samples})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    y_all = np.asarray([label_to_idx[s.label] for s in samples], dtype=np.int64)
    idx_all = np.arange(len(samples))

    idx_train_val, idx_test = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=seed,
        stratify=y_all,
    )

    y_train_val = y_all[idx_train_val]
    relative_train, relative_val = train_test_split(
        np.arange(len(idx_train_val)),
        test_size=val_size,
        random_state=seed,
        stratify=y_train_val,
    )

    idx_train = idx_train_val[relative_train]
    idx_val = idx_train_val[relative_val]

    train_samples = [samples[i] for i in idx_train]
    val_samples = [samples[i] for i in idx_val]
    test_samples = [samples[i] for i in idx_test]

    X_train, y_train = _to_arrays(train_samples, sequence_length, label_to_idx)
    X_val, y_val = _to_arrays(val_samples, sequence_length, label_to_idx)
    X_test, y_test = _to_arrays(test_samples, sequence_length, label_to_idx)

    # Augment only training split.
    X_train, y_train = _augment_split(X_train, y_train, copies=augment_copies, seed=seed)

    output_root.mkdir(parents=True, exist_ok=True)

    np.save(output_root / "train_X.npy", X_train)
    np.save(output_root / "train_y.npy", y_train)
    np.save(output_root / "val_X.npy", X_val)
    np.save(output_root / "val_y.npy", y_val)
    np.save(output_root / "test_X.npy", X_test)
    np.save(output_root / "test_y.npy", y_test)

    labels_payload = {
        "labels": labels,
        "label_to_idx": label_to_idx,
        "sequence_length": sequence_length,
    }
    (output_root / "labels.json").write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sequence_path", "label", "label_idx", "split"],
        )
        writer.writeheader()
        for split_name, split_samples in (
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ):
            for sample in split_samples:
                writer.writerow(
                    {
                        "sequence_path": str(sample.sequence_path),
                        "label": sample.label,
                        "label_idx": label_to_idx[sample.label],
                        "split": split_name,
                    }
                )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_labels": len(labels),
        "sequence_length": sequence_length,
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "augment_copies": augment_copies,
    }

    print("\n[Preprocessing] Dataset build complete")
    print(json.dumps(summary, indent=2))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train/val/test datasets from mp_data keypoint frames")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Root of mp_data-style keypoint directories")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Where to write processed dataset arrays")
    parser.add_argument("--sequence-length", type=int, default=30, help="Target sequence length")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation split ratio (from train_val)")
    parser.add_argument("--augment-copies", type=int, default=0, help="Number of augmented copies for train split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    build_dataset(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        sequence_length=args.sequence_length,
        test_size=args.test_size,
        val_size=args.val_size,
        augment_copies=args.augment_copies,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
