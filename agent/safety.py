"""L5 — Safety guardrails.

Guardrails wrap the reasoning core: they sanitize model-facing inputs
(basic prompt-injection defense) and clamp/validate model outputs so a
misbehaving or adversarial LLM response cannot produce an invalid or
unbounded RiskEvent.
"""
from __future__ import annotations

import re

from engines.types import RiskEvent

# Patterns that suggest an attempt to hijack the analyst prompt.
_INJECTION_PATTERNS = (
    r"ignore (?:all |the )?(?:previous|prior) instructions",
    r"system prompt",
    r"you are now",
    r"disregard (?:the )?(?:rules|guidelines)",
    r"developer mode",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

MAX_SUMMARY_LEN = 280


def sanitize_text(text: str) -> str:
    """Neutralize suspected injection phrases in free-text model inputs."""
    return _INJECTION_RE.sub("[redacted]", text)


def looks_like_injection(text: str) -> bool:
    """Return True if the text contains a known injection pattern."""
    return bool(_INJECTION_RE.search(text))


def enforce_output(event: RiskEvent) -> RiskEvent:
    """Clamp and bound a RiskEvent so downstream consumers can trust it."""
    event.risk = max(0.0, min(1.0, float(event.risk)))
    event.label = str(event.label)[:64].strip() or "unknown"
    summary = str(event.summary).strip()
    if len(summary) > MAX_SUMMARY_LEN:
        summary = summary[: MAX_SUMMARY_LEN - 1].rstrip() + "\u2026"
    event.summary = summary
    return event
