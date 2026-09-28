from __future__ import annotations

import base64
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np
from flask import Blueprint, Response, jsonify, request

from app.config.logging import get_logger
from app.config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()
KEYPOINT_FEATURES = 1662


def _require_cv2():
    """Lazy-import cv2 so Flask can boot without it installed."""
    try:
        import cv2 as _cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for media processing. "
            "Install it with: pip install opencv-python"
        ) from exc
    return _cv2


def _require_mediapipe():
    """Lazy-import mediapipe so Flask can boot without it installed."""
    try:
        import mediapipe as _mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for media processing. "
            "Install it with: pip install mediapipe"
        ) from exc
    return _mp

media_bp = Blueprint("media", __name__, url_prefix="/media")


class MediaProcessor:
    def __init__(self) -> None:
        mp = _require_mediapipe()
        self._mp_pose = mp.solutions.pose
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lock = threading.Lock()

    def process_frame(self, image: np.ndarray) -> dict[str, Any]:
        cv2 = _require_cv2()
        with self._lock:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = self._pose.process(rgb)
            hands_results = self._hands.process(rgb)

        left_hand, right_hand = self._split_hands(hands_results)
        keypoints = self._extract_keypoints(
            pose_results.pose_landmarks,
            left_hand,
            right_hand,
        )
        boxes = self._extract_boxes(
            pose_results.pose_landmarks,
            left_hand,
            right_hand,
            image.shape,
        )

        visualized = image.copy()
        self._draw_landmarks(visualized, pose_results.pose_landmarks, left_hand, right_hand)
        for label, (xmin, ymin, xmax, ymax) in boxes:
            cv2.rectangle(visualized, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                visualized,
                label,
                (xmin, max(ymin - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return {
            "visualized_image": visualized,
            "keypoints": keypoints,
            "boxes": [
                {
                    "label": label,
                    "bbox": {
                        "xmin": bbox[0],
                        "ymin": bbox[1],
                        "xmax": bbox[2],
                        "ymax": bbox[3],
                    },
                }
                for label, bbox in boxes
            ],
        }

    def _split_hands(self, hands_results: Any) -> tuple[Any | None, Any | None]:
        left_hand = None
        right_hand = None

        landmarks_list = getattr(hands_results, "multi_hand_landmarks", None) or []
        handedness_list = getattr(hands_results, "multi_handedness", None) or []

        for idx, hand_landmarks in enumerate(landmarks_list):
            label = ""
            if idx < len(handedness_list) and handedness_list[idx].classification:
                label = handedness_list[idx].classification[0].label.lower()

            if label == "left" and left_hand is None:
                left_hand = hand_landmarks
            elif label == "right" and right_hand is None:
                right_hand = hand_landmarks
            elif left_hand is None:
                left_hand = hand_landmarks
            elif right_hand is None:
                right_hand = hand_landmarks

        return left_hand, right_hand

    def _draw_landmarks(
        self,
        image: np.ndarray,
        pose_landmarks: Any,
        left_hand_landmarks: Any,
        right_hand_landmarks: Any,
    ) -> None:
        if pose_landmarks is not None:
            self._mp_drawing.draw_landmarks(
                image,
                pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
            )
        if left_hand_landmarks is not None:
            self._mp_drawing.draw_landmarks(
                image,
                left_hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
            )
        if right_hand_landmarks is not None:
            self._mp_drawing.draw_landmarks(
                image,
                right_hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
            )

    def _extract_keypoints(
        self,
        pose_landmarks: Any,
        left_hand_landmarks: Any,
        right_hand_landmarks: Any,
    ) -> list[float]:
        # Single source of truth: 1662-d WEBCAM layout, zero face block
        # (Pose+Hands live path). See app/services/keypoint_service.py.
        from app.services.keypoint_service import assemble_webcam_layout_no_face

        return assemble_webcam_layout_no_face(
            pose_landmarks, left_hand_landmarks, right_hand_landmarks
        )

    def _extract_boxes(
        self,
        pose_landmarks: Any,
        left_hand_landmarks: Any,
        right_hand_landmarks: Any,
        frame_shape: tuple[int, int, int],
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        frame_h, frame_w = frame_shape[:2]
        boxes: list[tuple[str, tuple[int, int, int, int]]] = []

        def get_bbox(landmarks: Any) -> tuple[int, int, int, int] | None:
            if not landmarks:
                return None

            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            xmin = int(min(xs) * frame_w)
            xmax = int(max(xs) * frame_w)
            ymin = int(min(ys) * frame_h)
            ymax = int(max(ys) * frame_h)
            pad = 20
            return (
                max(0, xmin - pad),
                max(0, ymin - pad),
                min(frame_w, xmax + pad),
                min(frame_h, ymax + pad),
            )

        candidates = (
            ("Left Hand", left_hand_landmarks),
            ("Right Hand", right_hand_landmarks),
            ("Pose", pose_landmarks),
        )

        for label, landmarks in candidates:
            bbox = get_bbox(landmarks)
            if bbox is not None:
                boxes.append((label, bbox))

        return boxes


class FallbackMediaProcessor:
    """Safe fallback when MediaPipe cannot be initialized in the environment."""

    def process_frame(self, image: np.ndarray) -> dict[str, Any]:
        return {
            "visualized_image": image,
            "keypoints": [0.0] * KEYPOINT_FEATURES,
            "boxes": [],
        }


media_processor: Any | None = None
media_processor_error: str | None = None


def _get_media_processor() -> Any:
    global media_processor
    global media_processor_error
    if media_processor is None:
        try:
            media_processor = MediaProcessor()
            media_processor_error = None
        except Exception as exc:
            logger.exception("Failed to initialize MediaProcessor; using fallback processor")
            media_processor_error = "Media processor unavailable; using fallback keypoints."
            media_processor = FallbackMediaProcessor()  # type: ignore[assignment]
    return media_processor


def _decode_data_url_to_bgr(data_url: str) -> np.ndarray:
    cv2 = _require_cv2()
    if not data_url:
        raise ValueError("Missing frame data")

    if "," not in data_url:
        raise ValueError("Invalid frame payload format")

    encoded = data_url.split(",", maxsplit=1)[1]
    raw = base64.b64decode(encoded)
    image_np = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Failed to decode image")

    return frame


def _encode_bgr_to_data_url(frame: np.ndarray) -> str:
    cv2 = _require_cv2()
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode image")

    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@media_bp.get("/health")
def media_health() -> tuple[Response, int]:
    _get_media_processor()
    payload = {
        "status": "ok" if media_processor_error is None else "degraded",
        "service": "media",
        "processor": "mediapipe_pose_hands" if media_processor_error is None else "fallback",
        "warning": media_processor_error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(payload), 200


@media_bp.post("/frame")
def process_frame_endpoint() -> tuple[Response, int]:
    try:
        payload = request.get_json(silent=True) or {}
        frame_data = payload.get("frame")
        if not frame_data:
            return jsonify({"error": "Field 'frame' is required"}), 400

        frame = _decode_data_url_to_bgr(frame_data)
        result = _get_media_processor().process_frame(frame)

        response = {
            "visualization": _encode_bgr_to_data_url(result["visualized_image"]),
            "keypoints": result["keypoints"],
            "boxes": result["boxes"],
            "warning": media_processor_error,
            "frame_size": {
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return jsonify(response), 200

    except ValueError as exc:
        logger.warning("Bad media frame request: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Media frame processing failed")
        return jsonify({"error": f"Media processing failed: {exc}"}), 500


@media_bp.get("/video_feed")
def video_feed() -> Response:
    try:
        cv2 = _require_cv2()
    except ImportError as exc:
        return jsonify({"error": str(exc)}), 503
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        return jsonify({"error": "Could not open webcam"}), 500

    def frame_generator() -> Any:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                processed = _get_media_processor().process_frame(frame)
                ok_jpg, buffer = cv2.imencode(".jpg", processed["visualized_image"])
                if not ok_jpg:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        finally:
            cap.release()

    return Response(
        frame_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def register_media_controller(flask_app: Any) -> None:
    flask_app.register_blueprint(media_bp)
