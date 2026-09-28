"""
app/services/inference_backends/torch_backend.py
──────────────────────────────────────────────────
PyTorch inference backend (generic).

Loads a .pth state-dict into the arch recorded in the registry entry
(lstm|gru|cnn_lstm|tcn|transformer) and runs prediction
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
        Load a PyTorch .pth state-dict into the arch recorded in the registry.

        Args:
            model_config: Registry entry containing:
                - model_path (str): path to .pth file
                - labels (list[str]): ordered label list
                - arch (str): one of lstm|gru|cnn_lstm|tcn|transformer
                  (absent = legacy transformer_v1 entry)
                - hyperparams (dict, optional): hidden_size/num_layers/
                  dropout/nhead used at training time
        """
        from ml.training.arch_factory import ARCH_DEFAULTS, build_model

        torch = _require_torch()
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path: str = model_config["model_path"]
        self._labels = list(model_config["labels"])

        arch = str(model_config.get("arch") or "transformer").lower()
        hyper = dict(model_config.get("hyperparams") or {})
        arch_defaults = ARCH_DEFAULTS.get(arch, ARCH_DEFAULTS["transformer"])

        logger.info("TorchBackend: loading arch=%s model from %s (device=%s)",
                    arch, model_path, self._device)

        model = build_model(
            arch,
            hidden_size=int(hyper.get("hidden_size", arch_defaults["hidden_size"])),
            num_layers=int(hyper.get("num_layers", arch_defaults["num_layers"])),
            dropout=float(hyper.get("dropout", arch_defaults["dropout"])),
            nhead=int(hyper.get("nhead", 8)),
            num_classes=len(self._labels),
            input_features=int(model_config.get("input_features", 1629)),
            labels=self._labels,
        ).to(self._device)
        state = torch.load(model_path, map_location=self._device, weights_only=True)
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
