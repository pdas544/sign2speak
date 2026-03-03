from __future__ import annotations

import base64
import threading
from datetime import datetime, timezone
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from flask import Blueprint, Response, jsonify, request

from app.config.logging import get_logger
from app.config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()

media_bp = Blueprint("media", __name__, url_prefix="/media")


class MediaProcessor:
    def __init__(self) -> None:
        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lock = threading.Lock()

    def process_frame(self, image: np.ndarray) -> dict[str, Any]:
        with self._lock:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self._holistic.process(rgb)

        keypoints = self._extract_keypoints(results)
        boxes = self._extract_boxes(results, image.shape)

        visualized = image.copy()
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

    def _extract_keypoints(self, results: Any) -> list[float]:
        def extract(landmarks: Any, count: int, dims: int) -> np.ndarray:
            if not landmarks:
                return np.zeros(count * dims, dtype=np.float32)

            if dims == 3:
                return np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
                ).flatten()

            return np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark],
                dtype=np.float32,
            ).flatten()

        pose = extract(results.pose_landmarks, 33, 4)
        face = extract(results.face_landmarks, 468, 3)
        left_hand = extract(results.left_hand_landmarks, 21, 3)
        right_hand = extract(results.right_hand_landmarks, 21, 3)

        return np.concatenate([pose, face, left_hand, right_hand]).tolist()

    def _extract_boxes(self, results: Any, frame_shape: tuple[int, int, int]) -> list[tuple[str, tuple[int, int, int, int]]]:
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
            ("Left Hand", results.left_hand_landmarks),
            ("Right Hand", results.right_hand_landmarks),
            ("Pose", results.pose_landmarks),
        )

        for label, landmarks in candidates:
            bbox = get_bbox(landmarks)
            if bbox is not None:
                boxes.append((label, bbox))

        return boxes


media_processor = MediaProcessor()


def _decode_data_url_to_bgr(data_url: str) -> np.ndarray:
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
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode image")

    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@media_bp.get("/health")
def media_health() -> tuple[Response, int]:
    payload = {
        "status": "ok",
        "service": "media",
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
        result = media_processor.process_frame(frame)

        response = {
            "visualization": _encode_bgr_to_data_url(result["visualized_image"]),
            "keypoints": result["keypoints"],
            "boxes": result["boxes"],
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
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        return jsonify({"error": "Could not open webcam"}), 500

    def frame_generator() -> Any:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                processed = media_processor.process_frame(frame)
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
