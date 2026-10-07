from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np

from app.config.logging import get_logger
from app.config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()


def _require_cv2():
    try:
        import cv2 as _cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for capture service. "
            "Install it with: pip install opencv-python"
        ) from exc
    return _cv2


class CaptureService:
    def __init__(self, camera_index: int | None = None, buffer_size: int | None = None) -> None:
        self._camera_index = settings.camera_index if camera_index is None else camera_index
        self._buffer_size = settings.max_seq_length if buffer_size is None else buffer_size

        self._buffer: deque[np.ndarray] = deque(maxlen=self._buffer_size)
        self._buffer_lock = threading.Lock()

        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return

        cv2 = _require_cv2()
        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self._camera_index}")

        self._capture = capture
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Capture service started with camera index %s", self._camera_index)

    def stop(self) -> None:
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        logger.info("Capture service stopped")

    def _capture_loop(self) -> None:
        assert self._capture is not None

        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.02)
                continue

            with self._buffer_lock:
                self._buffer.append(frame.copy())

    def get_latest_frame(self) -> np.ndarray | None:
        with self._buffer_lock:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()

    def get_recent_frames(self, count: int | None = None) -> list[np.ndarray]:
        with self._buffer_lock:
            if count is None or count >= len(self._buffer):
                return [frame.copy() for frame in self._buffer]
            return [frame.copy() for frame in list(self._buffer)[-count:]]

    def clear_buffer(self) -> None:
        with self._buffer_lock:
            self._buffer.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        with self._buffer_lock:
            return len(self._buffer)

    def stream_mjpeg(self, frame_transform: Any | None = None):
        cv2 = _require_cv2()
        while True:
            frame = self.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            if frame_transform is not None:
                try:
                    frame = frame_transform(frame)
                except Exception:
                    logger.warning("Frame transform failed", exc_info=True)

            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
