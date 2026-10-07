from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_dict(self) -> dict[str, int]:
        return {"xmin": self.xmin, "ymin": self.ymin, "xmax": self.xmax, "ymax": self.ymax}


@dataclass
class DetectedBox:
    label: str
    bbox: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "bbox": self.bbox.to_dict()}


@dataclass
class FrameResponse:
    visualization: str  # base64 data URL
    keypoints: list[float]
    boxes: list[DetectedBox]
    frame_size: dict[str, int]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "visualization": self.visualization,
            "keypoints": self.keypoints,
            "boxes": [b.to_dict() for b in self.boxes],
            "frame_size": self.frame_size,
            "timestamp": self.timestamp,
        }


@dataclass
class AudioOutput:
    en: str | None
    hi: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {"en": self.en, "hi": self.hi}


@dataclass
class PredictResponse:
    predicted_gloss: str
    confidence: float
    confidence_percent: float
    threshold: float
    accepted: bool
    audio: AudioOutput
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_gloss": self.predicted_gloss,
            "confidence": self.confidence,
            "confidence_percent": self.confidence_percent,
            "threshold": self.threshold,
            "accepted": self.accepted,
            "audio": self.audio.to_dict(),
            "timestamp": self.timestamp,
        }


@dataclass
class HealthResponse:
    status: str
    app: str
    environment: str
    timestamp: str
    dependencies: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "app": self.app,
            "environment": self.environment,
            "timestamp": self.timestamp,
            "dependencies": self.dependencies,
        }


@dataclass
class ErrorResponse:
    error: str
    timestamp: str

    def to_dict(self) -> dict[str, str]:
        return {"error": self.error, "timestamp": self.timestamp}
