"""L4 — HTTP request/response schemas.

Pydantic models define the wire contract for the API and provide converters
to and from the L3 engine dataclasses. Keeping the boundary explicit means the
internal `engines.types` can evolve without breaking external clients.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from engines.types import Detection, ValidatedDetections


class DetectionIn(BaseModel):
  """A single detected object supplied by the caller."""

  label: str
  confidence: float = Field(ge=0.0, le=1.0)
  bbox: tuple[float, float, float, float]
  track_id: int | None = None
  text: str | None = None

  def to_domain(self) -> Detection:
    return Detection(
      label=self.label,
      confidence=self.confidence,
      bbox=self.bbox,
      track_id=self.track_id,
      text=self.text,
    )


class AnalyzeRequest(BaseModel):
  """Validated detections for one frame plus optional free-text context."""

  frame_index: int = 0
  detections: list[DetectionIn] = Field(default_factory=list)
  dropped: int = 0
  note: str | None = None

  def to_domain(self) -> ValidatedDetections:
    return ValidatedDetections(
      frame_index=self.frame_index,
      items=[d.to_domain() for d in self.detections],
      dropped=self.dropped,
    )

  def context(self) -> dict:
    ctx: dict = {"frame_index": self.frame_index}
    if self.note is not None:
      ctx["note"] = self.note
    return ctx


class RiskEventOut(BaseModel):
  """The reasoning core's risk-scored output."""

  frame_index: int
  risk: float
  label: str
  summary: str


class AnalyzeResponse(BaseModel):
  """Envelope returned by the analyze endpoint."""

  event: RiskEventOut
  sanitized: bool
  used_fallback: bool


class HealthResponse(BaseModel):
  status: str = "ok"
  reasoning_backend: str
