"""Perception: turn one image into "what is in the picture".

    base.py       the Detector interface
    yolo.py       local YOLO (detection + pose), for a real camera
    synthetic.py  renders and decodes the simulated picture, for when there is no camera / model
    null.py       the null detector: every detection-based constraint degrades to a text prompt

build_detector() picks the one for real cameras from the settings; the simulated picture always uses SyntheticDetector,
assigned by source type in wiring.py.
"""
from __future__ import annotations

import logging

from ..config import Settings
from .base import Detector
from .null import NullDetector

log = logging.getLogger(__name__)


def build_detector(settings: Settings) -> Detector:
    mode = settings.detector
    if mode == "none":
        return NullDetector()
    if mode not in ("auto", "yolo"):
        raise ValueError(f"Unknown detector: {mode!r} (choose auto / yolo / none)")

    try:
        from .yolo import YoloDetector

        return YoloDetector(
            model=settings.yolo_model,
            pose_model=settings.yolo_pose_model,
            label_map=settings.label_map,
            conf=settings.yolo_conf,
            imgsz=settings.yolo_imgsz,
            device=settings.yolo_device,
        )
    except Exception as exc:
        missing = isinstance(exc, ImportError)
        reason = "ultralytics is not installed (pip install ultralytics)" if missing else f"YOLO failed to load: {exc}"
        if mode == "yolo":
            raise RuntimeError(reason) from exc
        # auto: do not keep the whole server from starting, but make it visible in both the log and the UI
        log.warning("%s -> no detection model loaded; detection-based constraints get text prompts only", reason)
        return NullDetector(note=f"{reason}; detection-based constraints get text prompts only")
