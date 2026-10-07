"""
app/services/inference_backends/tf_backend.py
──────────────────────────────────────────────
TensorFlow / Keras inference backend.

Loads a .h5 model (e.g. models/action_model_cnn_lstm_new.h5) and runs
prediction on a (sequence_length, input_features) keypoint array.

This is the original production path used in realtime_prediction.py.
"""

from __future__ import annotations

import numpy as np

from app.config.logging import get_logger
from app.services.inference_backends.base import InferenceBackend


logger = get_logger(__name__)


class TFBackend(InferenceBackend):
    """TensorFlow/Keras CNN+LSTM backend."""

    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] = []
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load(self, model_config: dict) -> None:
        """
        Load a Keras .h5 model.

        Args:
            model_config: Registry entry containing:
                - model_path (str): path to .h5 file
                - labels (list[str]): ordered label list
        """
        try:
            import tensorflow as tf  # lazy import – not always installed
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for the TF backend. "
                "Install it with: pip install tensorflow"
            ) from exc

        model_path: str = model_config["model_path"]
        self._labels = list(model_config["labels"])

        logger.info("TFBackend: loading model from %s", model_path)

        # Try standard deserialization first.
        # Some legacy .h5 models include `time_major` in LSTM configs,
        # which newer Keras versions reject.
        try:
            self._model = tf.keras.models.load_model(model_path, compile=False)
        except ValueError as exc:
            if "time_major" not in str(exc):
                raise

            logger.warning(
                "TFBackend: legacy LSTM config detected (time_major). "
                "Retrying model load with compatibility layer."
            )

            class LegacyCompatibleLSTM(tf.keras.layers.LSTM):
                def __init__(self, *args, **kwargs):
                    kwargs.pop("time_major", None)
                    super().__init__(*args, **kwargs)

            self._model = tf.keras.models.load_model(
                model_path,
                custom_objects={"LSTM": LegacyCompatibleLSTM},
                compile=False,
            )

        self._loaded = True
        logger.info("TFBackend: model loaded | labels=%d", len(self._labels))

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
            raise RuntimeError("TFBackend: model not loaded — call load() first")

        # Model expects (1, sequence_length, features)
        batch = np.expand_dims(keypoints, axis=0)
        probs = self._model.predict(batch, verbose=0)[0]

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        predicted_gloss = self._labels[idx]

        return predicted_gloss, confidence

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #

    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
