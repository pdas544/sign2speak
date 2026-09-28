"""
app/services/inference_backends/torch_backend.py
──────────────────────────────────────────────────
PyTorch Transformer inference backend.

Loads a .pth state-dict into SignLanguageTransformer and runs prediction
on a (sequence_length, input_features) keypoint array.
"""

from __future__ import annotations

import numpy as np

from app.config.logging import get_logger
from app.services.inference_backends.base import InferenceBackend


logger = get_logger(__name__)


def _require_torch():
    """Lazy-import torch so Flask can boot without it installed."""
    try:
        import torch as _torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the Torch backend. "
            "Install it with: pip install torch"
        ) from exc
    return _torch


class TorchBackend(InferenceBackend):
    """PyTorch Transformer backend."""

    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] = []
        self._torch = None
        self._device = "cpu"
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load(self, model_config: dict) -> None:
        """
        Load a PyTorch .pth state-dict into SignLanguageTransformer.

        Args:
            model_config: Registry entry containing:
                - model_path (str): path to .pth file
                - labels (list[str]): ordered label list
        """
        try:
            from model_transformer import Config, SignLanguageTransformer
        except ImportError as exc:
            raise ImportError(
                "model_transformer.py must be importable from the project root"
            ) from exc

        torch = _require_torch()
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path: str = model_config["model_path"]
        self._labels = list(model_config["labels"])

        config = Config()
        # Override num_classes in case the registry label count differs from Config default
        config.num_classes = len(self._labels)
        config.gloss_to_idx = {g: i for i, g in enumerate(self._labels)}
        config.idx_to_gloss = {i: g for g, i in config.gloss_to_idx.items()}

        logger.info("TorchBackend: loading model from %s (device=%s)", model_path, self._device)

        model = SignLanguageTransformer(config).to(self._device)
        state = torch.load(model_path, map_location=self._device)
        model.load_state_dict(state)
        model.eval()

        self._model = model
        self._loaded = True
        logger.info("TorchBackend: model loaded | labels=%d", len(self._labels))

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, keypoints: np.ndarray) -> tuple[str, float]:
        """
        Args:
            keypoints: (sequence_length, input_features) float32 array.

        Returns:
            (predicted_gloss, confidence)
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("TorchBackend: model not loaded — call load() first")

        torch = self._torch or _require_torch()
        tensor = torch.from_numpy(keypoints).unsqueeze(0).float().to(self._device)

        with torch.no_grad():
            outputs = self._model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        predicted_gloss = self._labels[int(pred_idx.item())]
        confidence_value = float(confidence.item())

        return predicted_gloss, confidence_value

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #

    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
