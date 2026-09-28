from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from app.config.logging import get_logger
from app.config.settings import get_settings
from app.services.inference_backends.base import InferenceBackend


logger = get_logger(__name__)
settings = get_settings()


class InferenceService:
    """
    Framework-agnostic inference service.

    Loads the active model from the registry (models/registry/registry.json)
    and dispatches predict() / labels() to the appropriate backend
    (TFBackend for .h5 models, TorchBackend for .pth models).
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._backend: InferenceBackend | None = None
        self._backend_loaded = False
        self._active_model_name: str | None = None

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _load_backend_if_needed(self) -> None:
        """Double-checked locking: load backend once, lazily."""
        if self._backend_loaded and self._backend is not None:
            return

        with self._lock:
            if self._backend_loaded and self._backend is not None:
                return

            from app.services.model_registry_service import ModelRegistryService

            registry = ModelRegistryService()
            model_config = registry.active_model_config()
            self._active_model_name = registry.active_model_name()

            logger.info(
                "InferenceService: loading backend for model '%s' (framework=%s)",
                self._active_model_name,
                model_config.get("framework"),
            )

            self._backend = registry.build_backend(model_config)
            self._backend_loaded = True

    def _reset_backend(self) -> None:
        """Force-reload the backend on next call (e.g. after model switch)."""
        with self._lock:
            self._backend = None
            self._backend_loaded = False
            self._active_model_name = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def normalize_keypoints(
        self,
        keypoints_raw: list[list[float]] | np.ndarray,
        required_len: int | None = None,
    ) -> np.ndarray:
        keypoints_np = np.asarray(keypoints_raw, dtype=np.float32)

        if keypoints_np.ndim != 2:
            raise ValueError("Expected keypoints with shape (sequence_length, feature_dim)")

        seq_len = required_len or settings.max_seq_length
        if keypoints_np.shape[0] < seq_len:
            raise ValueError(f"Need at least {seq_len} frames, got {keypoints_np.shape[0]}")

        return keypoints_np[-seq_len:]

    def labels(self) -> list[str]:
        self._load_backend_if_needed()
        return self._backend.labels()  # type: ignore[union-attr]

    def predict(self, keypoints: np.ndarray) -> tuple[str, float]:
        """
        Run sign-language inference.

        Args:
            keypoints: Float32 array (sequence_length, input_features).

        Returns:
            (predicted_gloss, confidence)
        """
        self._load_backend_if_needed()
        if self._backend is None:
            raise RuntimeError("Inference backend is not initialized")

        from app.services.keypoint_service import serving_check
        from app.services.model_registry_service import ModelRegistryService

        # Fail loudly on layout mismatch (VIDEO_1629 vs WEBCAM_1662) instead
        # of silently truncating/padding. ValueError -> HTTP 400 upstream.
        # Face-masked models accept the full 1629-d frame; the backend
        # applies the declared mask (see serving_check).
        serving_check(
            int(keypoints.shape[1]),
            ModelRegistryService().active_model_config(),
        )

        return self._backend.predict(keypoints)

    def switch_model(self, model_name: str) -> None:
        """
        Switch to a different registered model at runtime.

        Persists the new active model to the registry and triggers a
        backend reload on the next predict() call.
        """
        from app.services.model_registry_service import ModelRegistryService

        registry = ModelRegistryService()
        registry.set_active_model(model_name)
        self._reset_backend()
        logger.info("InferenceService: model switched to '%s'", model_name)

    def active_model_info(self) -> dict[str, Any]:
        """Return metadata about the currently loaded backend."""
        self._load_backend_if_needed()
        from app.services.model_registry_service import ModelRegistryService

        registry = ModelRegistryService()
        try:
            config = registry.active_model_config()
        except Exception:
            config = {}
        return {
            "name": self._active_model_name,
            "framework": config.get("framework"),
            "display_name": config.get("display_name"),
            "labels_count": len(self._backend.labels()) if self._backend else 0,
        }

    def generate_audio(self, text: str) -> str | None:
        if not settings.tts_enabled:
            return None

        try:
            from tts_helper import TTSHelper
        except ImportError:
            logger.warning("tts_helper not importable; audio generation skipped")
            return None

        output_dir = Path(settings.audio_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_name = f"{text}_{timestamp}.wav"
        output_path = output_dir / file_name

        try:
            tts = TTSHelper()
            tts.engine.save_to_file(text, str(output_path))
            tts.engine.runAndWait()
            return file_name
        except Exception as exc:
            logger.warning("Audio generation failed for '%s': %s", text, exc)
            return None
