from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from flask import Blueprint, Response, jsonify, request, send_from_directory

from app.config.logging import get_logger
from app.config.settings import get_settings
from app.services.inference_service import InferenceService


logger = get_logger(__name__)
settings = get_settings()
MIN_TEMPORAL_DELTA = 0.006

inference_bp = Blueprint("inference", __name__, url_prefix="/inference")

# Module-level singleton — loaded lazily on first request
_inference_service = InferenceService()


# ─────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _parse_keypoints(payload: dict[str, Any]) -> np.ndarray:
    keypoints_raw = payload.get("keypoints") or payload.get("kp")
    if not keypoints_raw:
        raise ValueError("Field 'keypoints' is required")
    return _inference_service.normalize_keypoints(keypoints_raw)


def _try_generate_audio(text: str, should_generate: bool) -> dict[str, str | None]:
    if not should_generate or not settings.tts_enabled:
        return {"en": None, "hi": None}

    try:
        from app.services.tts_service import TTSService
        from app.services.translation_service import TranslationService

        translation_svc = TranslationService()
        tts_svc = TTSService(translation_svc)
        audio = tts_svc.generate_bilingual_audio(text)
        return audio
    except Exception as exc:
        logger.warning("Bilingual audio generation failed: %s", exc)
        return {"en": None, "hi": None}


# ─────────────────────────────────────────────────────────────────────────── #
# Routes                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

@inference_bp.get("/labels")
def get_labels() -> tuple[Response, int]:
    labels = _inference_service.labels()
    return jsonify({
        "labels": labels,
        "count": len(labels),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


@inference_bp.post("/predict")
def predict() -> tuple[Response, int]:
    try:
        payload = request.get_json(silent=True) or {}
        keypoints = _parse_keypoints(payload)
        keypoints_std = float(np.std(keypoints))
        nonzero_ratio = float(np.count_nonzero(keypoints) / keypoints.size)
        temporal_delta = float(np.mean(np.abs(np.diff(keypoints, axis=0)))) if keypoints.shape[0] > 1 else 0.0
        # Features layout: [pose(132), face(1404), left_hand(63), right_hand(63)]
        hand_slice = keypoints[:, -126:]
        hand_nonzero_ratio = float(np.count_nonzero(hand_slice) / hand_slice.size)

        if keypoints_std < 1e-4 or temporal_delta < MIN_TEMPORAL_DELTA or hand_nonzero_ratio < 0.01:
            logger.info(
                "Predict rejected | low signal quality shape=%s std=%.6f temporal_delta=%.6f(min=%.4f) hand_nonzero_ratio=%.4f",
                tuple(keypoints.shape),
                keypoints_std,
                temporal_delta,
                MIN_TEMPORAL_DELTA,
                hand_nonzero_ratio,
            )
            return jsonify({
                "error": "Low landmark activity detected. Keep both hands visible and perform a clearer sign before pressing Stop.",
                "quality": {
                    "std": round(keypoints_std, 6),
                    "temporal_delta": round(temporal_delta, 6),
                    "min_temporal_delta": MIN_TEMPORAL_DELTA,
                    "hand_nonzero_ratio": round(hand_nonzero_ratio, 4),
                },
            }), 422

        predicted_gloss, confidence = _inference_service.predict(keypoints)
        should_generate_audio = bool(payload.get("generate_audio", True))
        audio = _try_generate_audio(predicted_gloss, should_generate_audio)

        logger.info(
            "Predict success | gloss=%s confidence=%.4f accepted=%s shape=%s std=%.6f nonzero_ratio=%.4f temporal_delta=%.6f hand_nonzero_ratio=%.4f",
            predicted_gloss,
            confidence,
            confidence >= settings.prediction_threshold,
            tuple(keypoints.shape),
            keypoints_std,
            nonzero_ratio,
            temporal_delta,
            hand_nonzero_ratio,
        )

        return jsonify({
            "predicted_gloss": predicted_gloss,
            "confidence": round(confidence, 6),
            "confidence_percent": round(confidence * 100.0, 2),
            "threshold": settings.prediction_threshold,
            "accepted": confidence >= settings.prediction_threshold,
            "audio": audio,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 200

    except ValueError as exc:
        logger.warning("Invalid predict request: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Predict dependency missing: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@inference_bp.get("/audio/<path:file_name>")
def get_audio(file_name: str) -> Response:
    audio_dir = Path(settings.audio_output_dir)
    return send_from_directory(audio_dir, file_name, as_attachment=False)


@inference_bp.get("/models")
def list_models() -> tuple[Response, int]:
    """Return all models registered in models/registry/registry.json."""
    try:
        from app.services.model_registry_service import ModelRegistryService
        registry = ModelRegistryService()
        return jsonify({
            "models": registry.list_models(),
            "active_model": registry.active_model_name(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 200
    except Exception as exc:
        logger.exception("Failed to list models")
        return jsonify({"error": str(exc)}), 500


@inference_bp.post("/models/active")
def switch_active_model() -> tuple[Response, int]:
    """
    Switch the active model at runtime.

    Body: { "model_name": "<registry_key>" }
    """
    try:
        payload = request.get_json(silent=True) or {}
        model_name = payload.get("model_name", "").strip()
        if not model_name:
            return jsonify({"error": "'model_name' is required"}), 400

        _inference_service.switch_model(model_name)
        return jsonify({
            "message": f"Active model switched to '{model_name}'",
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 200
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to switch model")
        return jsonify({"error": str(exc)}), 500


@inference_bp.get("/models/active")
def get_active_model() -> tuple[Response, int]:
    """Return metadata about the currently loaded model."""
    try:
        info = _inference_service.active_model_info()
        return jsonify(info), 200
    except Exception as exc:
        logger.exception("Failed to get active model info")
        return jsonify({"error": str(exc)}), 500


def register_inference_controller(flask_app: Any) -> None:
    flask_app.register_blueprint(inference_bp)
    logger.info("Inference controller registered")
