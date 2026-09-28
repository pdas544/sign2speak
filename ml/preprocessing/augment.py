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
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compose a lightweight augmentation pipeline.

    Order: scale -> temporal shift -> noise.
    """
    gen = rng or np.random.default_rng()
    x = np.asarray(sequence, dtype=np.float32)
    x = random_scale(x, low=scale_low, high=scale_high, rng=gen)
    x = temporal_jitter(x, max_shift=max_time_shift, rng=gen)
    x = add_gaussian_noise(x, std=noise_std, rng=gen)
    return x.astype(np.float32)
