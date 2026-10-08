"""The null detector: detects nothing.

Use: get the "picture in, text prompt out" path running before YOLO is installed.
It declares no capability, so every detection-based constraint degrades to a text prompt automatically (see fast_loop.load).

Dependencies: contracts.
"""
from __future__ import annotations

from typing import Any

from ..contracts import Detection


class NullDetector:
    name = "none"
    capabilities: frozenset[str] = frozenset()
    roles = None    # detects nothing, so it puts no limit on the roles a shot list may use

    def __init__(self, note: str = "No detection model loaded: detection-based constraints get text prompts only"):
        self.note = note

    def detect(self, image: Any) -> list[Detection]:
        return []
