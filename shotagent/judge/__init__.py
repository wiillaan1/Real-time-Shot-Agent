"""Semantic judges (used by the slow loop).

    base.py     the Judge interface
    mock.py     fake: returns pass / fail
    cosmos.py   real: Cosmos3-Reason

Why the package is called judge and not cosmos: what the slow loop needs is the role "someone who can watch a take and
answer yes/no questions". Cosmos is one implementation of it. If the Cosmos endpoint cannot be reached from the laptop on site,
switching to another vision model only takes one more class implementing the Judge interface.
"""
from __future__ import annotations

import os

from ..config import Settings
from .base import Judge


def build_judge(settings: Settings) -> Judge:
    if settings.judge == "mock":
        from .mock import MockJudge

        return MockJudge(latency_s=settings.mock_latency_s)
    if settings.judge == "cosmos":
        from .cosmos import CosmosJudge

        return CosmosJudge(
            base_url=settings.cosmos_url,
            model=settings.cosmos_model,
            api_key=os.environ.get("COSMOS_API_KEY", ""),
            timeout_s=settings.cosmos_timeout_s,
            max_tokens=settings.cosmos_max_tokens,
        )
    raise ValueError(f"Unknown semantic judge backend: {settings.judge!r} (choose mock / cosmos)")
