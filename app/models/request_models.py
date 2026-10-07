from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameRequest:
    frame: str  # base64 data URL (data:image/jpeg;base64,...)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameRequest":
        frame = (data.get("frame") or "").strip()
        if not frame:
            raise ValueError("Field 'frame' is required")
        return cls(frame=frame)


@dataclass
class PredictRequest:
    keypoints: list[list[float]]
    generate_audio: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictRequest":
        raw = data.get("keypoints") or data.get("kp")
        if not raw:
            raise ValueError("Field 'keypoints' is required")
        if not isinstance(raw, list):
            raise ValueError("Field 'keypoints' must be a list")
        return cls(
            keypoints=raw,
            generate_audio=bool(data.get("generate_audio", True)),
        )
