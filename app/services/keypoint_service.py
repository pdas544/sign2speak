"""
app/services/keypoint_service.py
─────────────────────────────────
Single source of truth for keypoint vector layouts.

Two layouts exist in this repo (production-plan.md §2) and MUST NOT be mixed:

VIDEO_1629 (comparison / video lineage, 1629-d):
    [pose(33*3 xyz), left_hand(21*3), right_hand(21*3), face(468*3)]
    Produced by extract-keypoints-full.py and ml/training/datasets/video_keypoints.py.

WEBCAM_1662 (legacy serving lineage, 1662-d):
    [pose(33*4 xyzv), face(468*3), left_hand(21*3), right_hand(21*3)]
    Produced by test_action_recognition.py and the /media/frame live path.

Cutover to VIDEO_1629 serving happens only when a PyTorch v2 model becomes
the active model (production-plan.md Phase 7). Until then, serving stays on
WEBCAM_1662 and this module only guarantees both assemblers are shared,
documented, and shape-checked against the registry.
"""

from __future__ import annotations

import numpy as np

from app.config.logging import get_logger


logger = get_logger(__name__)

VIDEO_FEATURES = 1629
WEBCAM_FEATURES = 1662
SEQUENCE_LENGTH = 30

# Face-masked video layout: pose(99) + hands(126) = 225-d. Declared explicitly
# per-model via registry hyperparams {"mask_face": true}; never inferred.
NOFACE_VIDEO_FEATURES = 225
FACE_START_COL = 225

# WEBCAM_1662 block boundaries: [pose132 xyzv | face1404 | lh63 | rh63]
_POSE_N = 33
_FACE_N = 468
_HAND_N = 21


def webcam_to_video_frame(frame1662: np.ndarray) -> np.ndarray:
    """
    Convert one WEBCAM_1662 frame to VIDEO_1629.

    Exact where defined: pose xyz values are copied verbatim, visibility is
    dropped (video lineage never had it), blocks reordered
    [pose,face,lh,rh] -> [pose,lh,rh,face]. The reverse (1629 -> 1662) is
    NOT defined — visibility cannot be invented — and raises.
    """
    vec = np.asarray(frame1662, dtype=np.float32).reshape(-1)
    if vec.shape[0] != WEBCAM_FEATURES:
        raise ValueError(f"webcam frame must be {WEBCAM_FEATURES}-d, got {vec.shape}")
    pose = vec[0:_POSE_N * 4].reshape(_POSE_N, 4)[:, :3].reshape(-1)  # drop visibility
    face = vec[_POSE_N * 4:_POSE_N * 4 + _FACE_N * 3]
    lh_start = _POSE_N * 4 + _FACE_N * 3
    lh = vec[lh_start:lh_start + _HAND_N * 3]
    rh = vec[lh_start + _HAND_N * 3:lh_start + 2 * _HAND_N * 3]
    out = np.concatenate([pose, lh, rh, face])
    assert out.shape == (VIDEO_FEATURES,), f"converted shape {out.shape}"
    return out


def adapt_for_model(keypoints: np.ndarray, model_config: dict) -> np.ndarray:
    """
    Adapt a serving keypoint batch to what the active model consumes.

    - Exact dim match -> passthrough.
    - WEBCAM_1662 input + VIDEO_1629 model -> explicit documented conversion.
    - Face-masked model + full 1629-d frame -> declared column mask.
    - Anything else -> ValueError (HTTP 400 upstream), never silent truncation.
    """
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected keypoints with shape (sequence_length, feature_dim)")
    dim = int(arr.shape[1])
    expected = int(model_config.get("input_features", 0) or 0)
    name = model_config.get("display_name") or "active model"
    hyper = model_config.get("hyperparams") or {}
    masked = bool(hyper.get("mask_face", False))

    if masked:
        if dim == NOFACE_VIDEO_FEATURES:
            return arr
        if dim == VIDEO_FEATURES:
            logger.info("adapt: applying declared pose+hands mask for '%s'", name)
            return np.asarray(arr[:, :FACE_START_COL], dtype=np.float32)
        if dim == WEBCAM_FEATURES:
            logger.info("adapt: 1662->1629 conversion + declared mask for '%s'", name)
            conv = np.stack([webcam_to_video_frame(row) for row in arr])
            return np.asarray(conv[:, :FACE_START_COL], dtype=np.float32)
        raise ValueError(
            f"Keypoint dim mismatch: serving produced {dim} features but masked "
            f"model '{name}' needs a 1629-d (or pre-masked 225-d) frame."
        )

    if expected and dim == expected:
        return arr
    if expected == VIDEO_FEATURES and dim == WEBCAM_FEATURES:
        logger.info("adapt: 1662->1629 conversion for '%s'", name)
        return np.stack([webcam_to_video_frame(row) for row in arr])
    raise ValueError(
        f"Keypoint dim mismatch: serving produced {dim} features "
        f"but model '{name}' expects {expected}. "
        "Check the keypoint layout (VIDEO_1629 vs WEBCAM_1662) before predicting."
    )


def _flat(landmarks, count: int, dims: int) -> np.ndarray:
    """Flatten MediaPipe landmarks (or zeros when absent) to (count*dims,)."""
    if not landmarks:
        return np.zeros(count * dims, dtype=np.float32)
    pts = getattr(landmarks, "landmark", landmarks)
    if dims == 3:
        return np.array([[lm.x, lm.y, lm.z] for lm in pts], dtype=np.float32).flatten()
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in pts], dtype=np.float32
    ).flatten()


def assemble_video_layout(
    pose_landmarks=None,
    left_hand_landmarks=None,
    right_hand_landmarks=None,
    face_landmarks=None,
) -> list[float]:
    """Assemble a 1629-d VIDEO frame: [pose(99), lh(63), rh(63), face(1404)]."""
    pose = _flat(pose_landmarks, 33, 3)
    left = _flat(left_hand_landmarks, 21, 3)
    right = _flat(right_hand_landmarks, 21, 3)
    face = _flat(face_landmarks, 468, 3)
    vec = np.concatenate([pose, left, right, face])
    assert vec.shape == (VIDEO_FEATURES,), f"video layout shape {vec.shape}"
    return vec.tolist()


def assemble_webcam_layout(
    pose_landmarks=None,
    left_hand_landmarks=None,
    right_hand_landmarks=None,
    face_landmarks=None,
) -> list[float]:
    """Assemble a 1662-d WEBCAM frame: [pose(132), face(1404), lh(63), rh(63)]."""
    pose = _flat(pose_landmarks, 33, 4)
    face = _flat(face_landmarks, 468, 3)
    left = _flat(left_hand_landmarks, 21, 3)
    right = _flat(right_hand_landmarks, 21, 3)
    vec = np.concatenate([pose, face, left, right])
    assert vec.shape == (WEBCAM_FEATURES,), f"webcam layout shape {vec.shape}"
    return vec.tolist()


def assemble_webcam_layout_no_face(
    pose_landmarks=None,
    left_hand_landmarks=None,
    right_hand_landmarks=None,
) -> list[float]:
    """1662-d WEBCAM frame with zero face block (Pose+Hands live path)."""
    return assemble_webcam_layout(
        pose_landmarks, left_hand_landmarks, right_hand_landmarks, None
    )


def assert_compatible(feature_dim: int, model_config: dict) -> None:
    """
    Fail loudly when serving keypoints don't match the active model.

    Replaces the old silent truncate/pad behavior with an explicit error
    naming the model, its expected dims, and what serving produced.
    """
    expected = int(model_config.get("input_features", 0) or 0)
    if expected and feature_dim != expected:
        name = model_config.get("display_name") or model_config.get("name", "active model")
        raise ValueError(
            f"Keypoint dim mismatch: serving produced {feature_dim} features "
            f"but model '{name}' expects {expected}. "
            "Check the keypoint layout (VIDEO_1629 vs WEBCAM_1662) before predicting."
        )


def serving_check(feature_dim: int, model_config: dict) -> None:
    """
    Entry-point dim check for serving (InferenceService.predict).

    Face-masked models (registry hyperparams mask_face=true) accept the full
    1629-d video frame — the backend applies the declared column mask loudly
    (see apply_declared_mask). Anything else must match exactly.
    """
    hyper = model_config.get("hyperparams") or {}
    if hyper.get("mask_face"):
        if feature_dim != VIDEO_FEATURES:
            name = model_config.get("display_name") or "masked model"
            raise ValueError(
                f"Keypoint dim mismatch: serving produced {feature_dim} features "
                f"but masked model '{name}' expects the full {VIDEO_FEATURES}-d "
                f"video frame (mask to {NOFACE_VIDEO_FEATURES}-d is applied inside)."
            )
        return
    assert_compatible(feature_dim, model_config)


def apply_declared_mask(keypoints: np.ndarray, model_config: dict) -> np.ndarray:
    """Apply the registry-declared column mask (currently only pose+hands)."""
    hyper = model_config.get("hyperparams") or {}
    if hyper.get("mask_face"):
        logger.info(
            "Applying declared mask pose+hands: %s -> %s cols",
            keypoints.shape[1], NOFACE_VIDEO_FEATURES,
        )
        return np.asarray(keypoints[:, :FACE_START_COL], dtype=np.float32)
    return keypoints
