"""
app/services/inference_backends/base.py
────────────────────────────────────────
Abstract base class that every inference backend must implement.

A backend is responsible for:
  - Loading the model artifact from disk
  - Running inference on a normalised keypoint array
  - Reporting which labels it knows
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InferenceBackend(ABC):
    """Common interface for sign-language inference backends."""

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load(self, model_config: dict) -> None:
        """
        Load the model from disk given a model-registry entry dict.

        Args:
            model_config: One entry from models/registry/registry.json.
                          Required keys: model_path, labels, sequence_length.
        """

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def predict(self, keypoints: np.ndarray) -> tuple[str, float]:
        """
        Run inference on a normalised keypoint sequence.

        Args:
            keypoints: Float32 array of shape (sequence_length, input_features).

        Returns:
            (predicted_gloss, confidence)  where confidence ∈ [0, 1].
        """

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def labels(self) -> list[str]:
        """Return the list of gloss labels this backend was trained on."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True once load() has completed successfully."""
