from __future__ import annotations

import threading
from typing import Any

import numpy as np


def _require_cv2():
    try:
        import cv2 as _cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for media processing. "
            "Install it with: pip install opencv-python"
        ) from exc
    return _cv2


def _require_mediapipe():
    try:
        import mediapipe as _mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for media processing. "
            "Install it with: pip install mediapipe"
        ) from exc
    return _mp


class MediaService:
    def __init__(self) -> None:
        mp = _require_mediapipe()
        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lock = threading.Lock()

    def process_frame(self, image: np.ndarray) -> dict[str, Any]:
        cv2 = _require_cv2()
        with self._lock:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self._holistic.process(rgb)

        keypoints = self.extract_keypoints(results)
        boxes = self.extract_boxes(results, image.shape)

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

    def extract_keypoints(self, results: Any) -> list[float]:
        # Single source of truth: 1662-d WEBCAM layout.
        # See app/services/keypoint_service.py.
        from app.services.keypoint_service import assemble_webcam_layout

        return assemble_webcam_layout(
            results.pose_landmarks,
            results.left_hand_landmarks,
            results.right_hand_landmarks,
            results.face_landmarks,
        )

    def extract_boxes(
        self,
        results: Any,
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
            ("Left Hand", results.left_hand_landmarks),
            ("Right Hand", results.right_hand_landmarks),
            ("Pose", results.pose_landmarks),
        )

        for label, landmarks in candidates:
            bbox = get_bbox(landmarks)
            if bbox is not None:
                boxes.append((label, bbox))

        return boxes
