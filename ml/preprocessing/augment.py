"""
ml/preprocessing/augment.py
---------------------------
Reusable sequence-level augmentation helpers for sign-language keypoint data.

All functions operate on arrays shaped:
    (sequence_length, feature_dim)
"""

from __future__ import annotations

import numpy as np


def pad_or_truncate(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Pad (front) or truncate (keep recent frames) to a fixed sequence length."""
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {seq.shape}")

    if seq.shape[0] >= target_length:
        return seq[-target_length:]

    pad_rows = target_length - seq.shape[0]
    return np.pad(seq, ((pad_rows, 0), (0, 0)), mode="constant")


def add_gaussian_noise(sequence: np.ndarray, std: float = 0.01, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add small Gaussian perturbation to simulate sensor/capture noise."""
    gen = rng or np.random.default_rng()
    noise = gen.normal(loc=0.0, scale=std, size=sequence.shape).astype(np.float32)
    return (sequence + noise).astype(np.float32)


def random_scale(sequence: np.ndarray, low: float = 0.95, high: float = 1.05, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply uniform scale on all features to improve robustness."""
    gen = rng or np.random.default_rng()
    factor = float(gen.uniform(low, high))
    return (sequence * factor).astype(np.float32)


def _as_joint_triplets(sequence: np.ndarray) -> tuple[np.ndarray, int]:
    """Reshape (T, F) flat xyz rows to (T, J, 3); leftover cols untouched."""
    x = np.asarray(sequence, dtype=np.float32)
    n_joints = x.shape[1] // 3
    return x.reshape(x.shape[0], n_joints, 3), x.shape[1] - n_joints * 3


def rotate_sequence(sequence: np.ndarray, max_degrees: float = 7.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """In-plane rotation about the per-frame signer centroid (camera roll/tilt).

    One angle is sampled per SEQUENCE and applied to all frames so motion
    coherence is preserved. Small angles only: 2D rotation approximates true
    viewpoint change; large angles fabricate impossible skeletons.
    """
    gen = rng or np.random.default_rng()
    theta = np.radians(float(gen.uniform(-max_degrees, max_degrees)))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    x = np.asarray(sequence, dtype=np.float32).copy()
    pts, _ = _as_joint_triplets(x)
    xy = pts[:, :, :2]
    center = xy.mean(axis=1, keepdims=True)
    centered = xy - center
    pts[:, :, 0] = centered[..., 0] * cos_t - centered[..., 1] * sin_t + center[..., 0]
    pts[:, :, 1] = centered[..., 0] * sin_t + centered[..., 1] * cos_t + center[..., 1]
    return x.astype(np.float32)


def translate_sequence(sequence: np.ndarray, max_shift: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    """Uniform x/y shift per sequence (camera framing shifts)."""
    gen = rng or np.random.default_rng()
    dx = float(gen.uniform(-max_shift, max_shift))
    dy = float(gen.uniform(-max_shift, max_shift))
    x = np.asarray(sequence, dtype=np.float32).copy()
    pts, _ = _as_joint_triplets(x)
    pts[:, :, 0] += dx
    pts[:, :, 1] += dy
    return x.astype(np.float32)


def swap_hands(sequence: np.ndarray) -> np.ndarray:
    """Swap left/right hand blocks (handedness-invariance training).

    Layouts: 225 [pose99,lh63,rh63] | 1629 [pose99,lh63,rh63,face1404].
    Probe finding (Oct 2026): MediaPipe handedness is unstable run-to-run
    (same physical hand lands in L on one attempt, R on the next), so no
    static mirror toggle can fix it — the model must see both assignments.
    """
    x = np.asarray(sequence, dtype=np.float32).copy()
    if x.shape[1] in (225, 1629):
        x[:, 99:162], x[:, 162:225] = x[:, 162:225].copy(), x[:, 99:162].copy()
    return x


def temporal_jitter(sequence: np.ndarray, max_shift: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    """Shift a sequence in time with zero-padding, preserving shape."""
    gen = rng or np.random.default_rng()
    shift = int(gen.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return sequence.astype(np.float32)

    out = np.zeros_like(sequence, dtype=np.float32)
    if shift > 0:
        out[shift:] = sequence[:-shift]
    else:
        out[:shift] = sequence[-shift:]
    return out


def augment_sequence(
    sequence: np.ndarray,
    *,
    noise_std: float = 0.01,
    scale_low: float = 0.95,
    scale_high: float = 1.05,
    max_time_shift: int = 2,
    max_rotation_deg: float = 7.0,
    max_translate: float = 0.05,
    hand_swap_prob: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compose a lightweight augmentation pipeline.

    Order: scale -> rotate -> translate -> temporal shift -> noise,
    plus random left/right hand swap (handedness invariance).
    Geometric transforms use one sample per sequence (motion coherence).
    """
    gen = rng or np.random.default_rng()
    x = np.asarray(sequence, dtype=np.float32)
    x = random_scale(x, low=scale_low, high=scale_high, rng=gen)
    x = rotate_sequence(x, max_degrees=max_rotation_deg, rng=gen)
    x = translate_sequence(x, max_shift=max_translate, rng=gen)
    if float(gen.random()) < hand_swap_prob:
        x = swap_hands(x)
    x = temporal_jitter(x, max_shift=max_time_shift, rng=gen)
    x = add_gaussian_noise(x, std=noise_std, rng=gen)
    return x.astype(np.float32)
