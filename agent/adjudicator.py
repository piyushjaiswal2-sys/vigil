"""L5 — Adjudicator: orchestrate reasoning + safety into a final decision.

The Adjudicator is the single entry point the L4 server calls. It:
  1. sanitizes any free-text context before it reaches the reasoning core,
  2. asks the injected `ReasoningProvider` to score the validated detections,
  3. clamps the returned event so downstream consumers can trust its bounds.

This keeps the untrusted-input boundary and the output-contract boundary in
one place, independent of which LLM backend is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass

from engines.types import RiskEvent, ValidatedDetections
from agent.provider import ReasoningProvider, heuristic_risk
from agent.safety import enforce_output, looks_like_injection, sanitize_text


@dataclass
class Decision:
  event: RiskEvent
  sanitized: bool
  used_fallback: bool


class Adjudicator:
  """Turn validated detections into a trusted, risk-scored decision."""

  def __init__(self, provider: ReasoningProvider | None = None) -> None:
    self._provider = provider

  def decide(
    self, detections: ValidatedDetections, context: dict | None = None
  ) -> Decision:
    context = dict(context or {})
    sanitized = False
    note = context.get("note")
    if isinstance(note, str) and looks_like_injection(note):
      context["note"] = sanitize_text(note)
      sanitized = True

    used_fallback = self._provider is None
    if self._provider is None:
      event = RiskEvent(
        frame_index=detections.frame_index,
        risk=heuristic_risk(detections),
        label="heuristic",
        summary="No reasoning backend configured; used heuristic risk.",
        detections=detections.items,
        meta={"provider": "heuristic"},
      )
    else:
      event = self._provider.adjudicate(detections, context)

    event = enforce_output(event)
    return Decision(event=event, sanitized=sanitized, used_fallback=used_fallback)
