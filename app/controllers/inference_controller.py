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


def _min_motion_for(model_config: dict, feature_dim: int) -> float:
    """Motion gate threshold, calibrated per feature layout.

    Train-motion percentiles (mean abs frame diff): masked-225 uniform clips
    p1=0.0045/p5=0.008 (threshold 0.006 ~= p2); full-dim clips carry ~86%
    dead face dims, so the legacy 0.006 is kept unchanged there to avoid
    regressing the TF webcam path. Registry hyperparams min_temporal_delta
    overrides both.
    """
    override = (model_config.get("hyperparams") or {}).get("min_temporal_delta")
    if override is not None:
        return float(override)
    return 0.006


def _hand_block_stats(arr: np.ndarray) -> dict[str, float]:
    """Per-hand nonzero ratios for mirror-swap diagnosis.

    Layouts: 225 [pose99,lh63,rh63] | 1629 [pose99,lh63,rh63,face1404] |
    1662 [pose132,face1404,lh63,rh63]. Last 126 cols are always (lh, rh).
    """
    dim = int(arr.shape[1])

    def _nz(cols: np.ndarray) -> float:
        return float(np.count_nonzero(cols) / cols.size) if cols.size else 0.0

    if dim in (225, 1629, 1662):
        # frame-major safe split: last 126 cols = [lh(63) | rh(63)] per row
        lh = arr[:, arr.shape[1] - 126:arr.shape[1] - 63]
        rh = arr[:, arr.shape[1] - 63:]
        return {"hand_l": _nz(lh), "hand_r": _nz(rh), "hands": _nz(np.concatenate([lh, rh], axis=1))}
    return {"hand_l": 0.0, "hand_r": 0.0, "hands": _nz(arr[:, -126:])}

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

        # Adapt to the active model FIRST so quality metrics are computed on
        # the features the model actually sees (a 1662-d frame is 86% dead
        # face zeros for a masked 225-d model, which used to dilute motion
        # below the gate and reject legit signing).
        try:
            adapted, model_config = _inference_service.prepare(keypoints)
        except ValueError as exc:
            logger.warning("Predict dim mismatch: %s", exc)
            return jsonify({"error": str(exc)}), 400
        model_name = model_config.get("display_name") or "model"

        keypoints_std = float(np.std(adapted))
        nonzero_ratio = float(np.count_nonzero(adapted) / adapted.size)
        temporal_delta = float(np.mean(np.abs(np.diff(adapted, axis=0)))) if adapted.shape[0] > 1 else 0.0
        hand_stats = _hand_block_stats(adapted)
        hand_nonzero_ratio = hand_stats["hands"]
        min_motion = _min_motion_for(model_config, int(adapted.shape[1]))

        if keypoints_std < 1e-4 or temporal_delta < min_motion or hand_nonzero_ratio < 0.01:
            logger.info(
                "Predict rejected | model=%s raw=%s feat=%sd std=%.6f temporal_delta=%.6f(min=%.4f) hands=%.4f(L=%.3f,R=%.3f)",
                model_name,
                tuple(keypoints.shape),
                adapted.shape[1],
                keypoints_std,
                temporal_delta,
                min_motion,
                hand_nonzero_ratio,
                hand_stats["hand_l"],
                hand_stats["hand_r"],
            )
            return jsonify({
                "error": "Low landmark activity detected. Keep both hands visible and perform a clearer sign before pressing Stop.",
                "quality": {
                    "std": round(keypoints_std, 6),
                    "temporal_delta": round(temporal_delta, 6),
                    "min_temporal_delta": min_motion,
                    "hand_nonzero_ratio": round(hand_nonzero_ratio, 4),
                    "hand_l": round(hand_stats["hand_l"], 4),
                    "hand_r": round(hand_stats["hand_r"], 4),
                },
            }), 422

        predicted_gloss, confidence = _inference_service.predict(keypoints)
        should_generate_audio = bool(payload.get("generate_audio", True))
        audio = _try_generate_audio(predicted_gloss, should_generate_audio)

        logger.info(
            "Predict success | model=%s gloss=%s confidence=%.4f accepted=%s raw=%s feat=%sd std=%.6f nonzero_ratio=%.4f temporal_delta=%.6f hands=%.4f(L=%.3f,R=%.3f)",
            model_name,
            predicted_gloss,
            confidence,
            confidence >= settings.prediction_threshold,
            tuple(keypoints.shape),
            adapted.shape[1],
            keypoints_std,
            nonzero_ratio,
            temporal_delta,
            hand_nonzero_ratio,
            hand_stats["hand_l"],
            hand_stats["hand_r"],
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
