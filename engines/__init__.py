"""L3 — Vision engines package.

Importing this package registers the S0-S3 vision blocks into the block
registry so they can be referenced by name in a graph spec.
"""
from engines import blocks  # noqa: F401  (import for registration side effect)
from engines.types import (
    Detection,
    Detections,
    Frame,
    NormalizedFrame,
    RiskEvent,
    ValidatedDetections,
)

__all__ = [
    "Detection",
    "Detections",
    "Frame",
    "NormalizedFrame",
    "RiskEvent",
    "ValidatedDetections",
]
