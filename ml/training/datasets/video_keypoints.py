"""
ml/training/datasets/video_keypoints.py
----------------------------------------
Single shared loader for the video-derived keypoint dataset
(`processed/metadata.csv` + `processed/keypoints/*/*.pt`).

Canonical spec for the multi-model comparison (production-plan.md §2):
  - 15 video glosses, fixed sorted label order
  - sequence_length = 30 (most-recent frames, matches serving)
  - input_features = 1629 (pose xyz w/o visibility + hands + face)

Layout per frame (1629 = 99 + 63 + 63 + 1404):
  [pose(33*3), left_hand(21*3), right_hand(21*3), face(468*3)]

NOTE on metadata duplicates: `processed/metadata.csv` contains 695 rows but
only 531 unique `file_path` values ("skipped (already processed)" rows repeat
an already-saved file). This loader dedupes by resolved file path so no clip
is counted twice.

Usage:
    from ml.training.datasets.video_keypoints import build_arrays, create_dataloaders
    X_train, y_train, X_val, y_val, X_test, y_test, labels = build_arrays()
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ml.preprocessing.augment import pad_or_truncate


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA = PROJECT_ROOT / "processed" / "metadata.csv"

SEQUENCE_LENGTH = 30
INPUT_FEATURES = 1629

# Canonical label order for all v2 comparison models (sorted = deterministic).
VIDEO_LABELS: list[str] = [
    "beautiful", "big", "boy", "friend", "go", "good", "happy",
    "hello", "like", "nice", "no", "sister", "teacher", "what", "white",
]

# Metadata rows worth loading (everything else is a failed extraction).
USABLE_STATUSES = {"success", "skipped (already processed)"}


def _fix_features(seq: np.ndarray, input_features: int = INPUT_FEATURES) -> np.ndarray:
    """Truncate or zero-pad the feature axis to `input_features`."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    if seq.ndim != 2:
        seq = np.reshape(seq, (seq.shape[0], -1)).astype(np.float32)
    if seq.shape[1] > input_features:
        seq = seq[:, :input_features]
    elif seq.shape[1] < input_features:
        seq = np.pad(seq, ((0, 0), (0, input_features - seq.shape[1])), mode="constant")
    return seq.astype(np.float32)


def load_sequence(
    path: str | Path,
    sequence_length: int = SEQUENCE_LENGTH,
    input_features: int = INPUT_FEATURES,
) -> np.ndarray:
    """Load one clip (.pt or .npy) and normalize to (sequence_length, input_features)."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(path))
    elif suffix == ".pt":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required to load .pt keypoint files") from exc
        tensor = torch.load(str(path), map_location="cpu", weights_only=True)
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    else:
        raise ValueError(f"Unsupported feature file extension: {path.suffix}")

    arr = _fix_features(np.asarray(arr, dtype=np.float32), input_features)
    return pad_or_truncate(arr, target_length=sequence_length).astype(np.float32)


def read_split(
    metadata_path: str | Path = DEFAULT_METADATA,
    split: str = "train",
    labels: list[str] | None = None,
) -> tuple[list[Path], list[int], list[str]]:
    """
    Read (file_path, label_idx) pairs for one split, deduped by file path.

    Returns (paths, label_indices, labels).
    """
    labels = list(labels or VIDEO_LABELS)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    metadata_path = Path(metadata_path)

    paths: list[Path] = []
    indices: list[int] = []
    seen: set[str] = set()

    with metadata_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != split:
                continue
            if row.get("status") not in USABLE_STATUSES:
                continue
            label = (row.get("label") or "").strip()
            if label not in label_to_idx:
                continue
            raw_path = (row.get("file_path") or "").strip()
            if not raw_path or raw_path == "N/A":
                continue
            full = Path(raw_path)
            if not full.is_absolute():
                full = PROJECT_ROOT / full
            key = str(full.resolve()) if full.exists() else str(full)
            if key in seen or not full.exists():
                continue
            seen.add(key)
            paths.append(full)
            indices.append(label_to_idx[label])

    if not paths:
        raise RuntimeError(f"No usable '{split}' samples found in {metadata_path}")
    return paths, indices, labels


def build_arrays(
    metadata_path: str | Path = DEFAULT_METADATA,
    sequence_length: int = SEQUENCE_LENGTH,
    input_features: int = INPUT_FEATURES,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load all three splits as float32 arrays. Returns (Xtr, ytr, Xva, yva, Xte, yte, labels)."""
    labels = list(labels or VIDEO_LABELS)

    def _split_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
        paths, indices, _ = read_split(metadata_path, split, labels)
        X = np.stack([load_sequence(p, sequence_length, input_features) for p in paths])
        return X.astype(np.float32), np.asarray(indices, dtype=np.int64)

    X_train, y_train = _split_arrays("train")
    X_val, y_val = _split_arrays("val")
    X_test, y_test = _split_arrays("test")
    return X_train, y_train, X_val, y_val, X_test, y_test, labels


# --------------------------------------------------------------------------- #
# PyTorch Dataset / DataLoaders (torch imported lazily)                       #
# --------------------------------------------------------------------------- #

class VideoKeypointDataset:
    """Fixed-length (30, 1629) torch Dataset over one metadata split."""

    def __init__(
        self,
        metadata_path: str | Path = DEFAULT_METADATA,
        split: str = "train",
        labels: list[str] | None = None,
        sequence_length: int = SEQUENCE_LENGTH,
        input_features: int = INPUT_FEATURES,
    ) -> None:
        try:
            import torch
            from torch.utils.data import Dataset as _TorchDataset
        except ImportError as exc:
            raise ImportError("PyTorch is required for VideoKeypointDataset") from exc
        self._torch = torch
        self._base = _TorchDataset
        self.sequence_length = sequence_length
        self.input_features = input_features
        self.labels = list(labels or VIDEO_LABELS)
        self.paths, indices, _ = read_split(metadata_path, split, self.labels)
        self.targets = list(indices)
        self.split = split

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        seq = load_sequence(self.paths[idx], self.sequence_length, self.input_features)
        return (
            self._torch.from_numpy(seq),
            self._torch.tensor(self.targets[idx], dtype=self._torch.long),
        )


def create_dataloaders(
    metadata_path: str | Path = DEFAULT_METADATA,
    batch_size: int = 16,
    labels: list[str] | None = None,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    """Return (train_loader, val_loader, test_loader) over fixed-length clips."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required for create_dataloaders") from exc
    labels = list(labels or VIDEO_LABELS)
    train_ds = VideoKeypointDataset(metadata_path, "train", labels)
    val_ds = VideoKeypointDataset(metadata_path, "val", labels)
    test_ds = VideoKeypointDataset(metadata_path, "test", labels)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
