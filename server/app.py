"""L4 — FastAPI application factory.

`create_app()` builds the ASGI app, selects a reasoning backend from the
environment (freellmapi when configured, heuristic fallback otherwise), and
injects a single shared `Adjudicator` into the router via a dependency
override. This keeps route handlers pure and easy to unit-test.
"""
from __future__ import annotations

import os

from fastapi import FastAPI

from agent import Adjudicator, FreeLlmApiProvider
from server.routes import get_adjudicator, router


def build_adjudicator() -> Adjudicator:
  """Wire the reasoning backend based on environment configuration."""
  if os.getenv("FREELLMAPI_BASE_URL"):
    return Adjudicator(provider=FreeLlmApiProvider())
  return Adjudicator(provider=None)


def create_app() -> FastAPI:
  app = FastAPI(
    title="VIGIL",
    version="0.1.0",
    description="Vision + reasoning pipeline API (L4 serving layer).",
  )

  adjudicator = build_adjudicator()
  app.dependency_overrides[get_adjudicator] = lambda: adjudicator
  app.include_router(router, prefix="/v1")

  @app.get("/", tags=["ops"])
  def root() -> dict:
    return {"service": "vigil", "docs": "/docs", "api": "/v1"}

  return app


app = create_app()
