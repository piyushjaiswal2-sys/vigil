"""L3 — Real detector backends for the DetectBlock.

The DetectBlock (S2) stays weight-free and importable by accepting an
injected `detector` callable via its config. This module provides the real
YOLOv8 backend that callable expects, adapting `ultralytics` results into
VIGIL's dependency-light `Detection` contract.

Usage:

    from engines.detectors import YoloV8Detector
    det = YoloV8Detector(model="yolov8n.pt", conf=0.25)
    block = DetectBlock(config={"detector": det})

`ultralytics`/`torch` are imported lazily inside `_ensure_model`, so simply
importing this module never pulls heavy dependencies — the graph stays
testable on machines without them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engines.types import Detection, NormalizedFrame


@dataclass
class YoloV8Detector:
    """Adapt an ultralytics YOLOv8 model to the VIGIL detector callable.

    Called with a `NormalizedFrame` whose `.data` is a numpy image, an image
    path, or a PIL image (anything ultralytics accepts). Returns a list of
    `Detection` with pixel-space xyxy boxes so downstream IoU/metrics match
    the source image resolution.
    """

    model: str = "yolov8n.pt"
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str = "cpu"
    _model: Any = field(default=None, init=False, repr=False)
    _names: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO  # lazy: keeps module import light

        self._model = YOLO(self.model)
        self._names = dict(self._model.names)

    def _source(self, frame: NormalizedFrame | Any) -> Any:
        """Extract an ultralytics-consumable source from a frame or raw input."""
        if isinstance(frame, NormalizedFrame):
            return frame.data if frame.data is not None else frame.source
        return frame  # allow passing a path / array / PIL image directly

    def __call__(self, frame: NormalizedFrame | Any) -> list[Detection]:
        source = self._source(frame)
        if source is None:
            return []
        self._ensure_model()
        results = self._model.predict(
            source=source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=self._names.get(cls_id, str(cls_id)),
                        confidence=float(box.conf[0]),
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections
