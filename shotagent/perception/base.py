"""The detector interface: one image in, a list of detections out.

The fast loop knows only this interface and does not care whether a local YOLO, the organisers' remote endpoint
or the simulated picture's decoder is behind it. To add a detection backend (say the organisers' $YOLO_URL), implement
this interface and add a branch to build_detector() in perception/__init__.py. The fast loop does not change by a line.

Dependencies: contracts only.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

from ..contracts import Detection


class Detector(Protocol):
    name: str                        # shown in the UI's status line
    capabilities: frozenset[str]     # a subset of contracts.CAP_DETECT / CAP_POSE
    note: str                        # anything the user should know (e.g. "ultralytics missing, degraded")
    roles: Optional[frozenset[str]]  # which roles it recognises; None = cannot say. Generated shot lists are limited to these

    def detect(self, image: Any) -> list[Detection]:
        """image is a BGR numpy array. Returned boxes and keypoints are normalised to [0,1]."""
        ...
