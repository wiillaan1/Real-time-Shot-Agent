"""Local YOLO detector (ultralytics): the detection model gives boxes, the pose model gives a person's keypoints.

Why the fast loop runs a local model instead of the organisers' $YOLO_URL: their endpoint takes a base64 video clip
(POST /v1/infer, see .cursor/skills/gpu/README.md in the starter repo). It is meant for finished clips, and going over
the network per frame can hardly stay under a second. Local is fast enough: yolo11n + yolo11n-pose together take about
0.2 s per frame on a 2-core CPU test machine (measure your own laptop with scripts/check_yolo.py).

The class name -> role name mapping is config.DEFAULT_LABEL_MAP:
    backpack / handbag / suitcase -> bag, dining table -> table, person -> person, ...

Dependencies: contracts. ultralytics is imported at construction, so the other detectors work without it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..contracts import CAP_DETECT, CAP_POSE, Box, Detection

# The fixed order of COCO human keypoints (the output order of ultralytics pose models)
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class YoloDetector:
    note = ""

    def __init__(
        self,
        model: str = "yolo11n.pt",
        pose_model: str = "yolo11n-pose.pt",
        label_map: Mapping[str, str] | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "",
    ):
        from ultralytics import YOLO   # raises ImportError when missing; build_detector handles it

        self._label_map = dict(label_map or {})
        self._conf = conf
        self._imgsz = imgsz
        self._device = device or None

        self._det = YOLO(model)
        self._pose = YOLO(pose_model) if pose_model else None

        # With a pose model, people come from it (box + keypoints), so the detection model does not report them again
        wanted = {raw for raw in self._label_map if not (self._pose is not None and raw == "person")}
        self._class_ids = [i for i, raw in self._det.names.items() if raw in wanted]

        self.capabilities = frozenset({CAP_DETECT} | ({CAP_POSE} if self._pose is not None else set()))
        self.roles = frozenset(self._label_map.values())
        self.name = "yolo(" + Path(model).stem + (" + " + Path(pose_model).stem if self._pose else "") + ")"

        # The first inference initialises things and is much slower: run one blank frame at startup so the first real frame does not pay for it
        self.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    def detect(self, image: Any) -> list[Detection]:
        found: list[Detection] = []

        if self._class_ids:
            result = self._det.predict(
                image, conf=self._conf, imgsz=self._imgsz, classes=self._class_ids,
                device=self._device, verbose=False,
            )[0]
            boxes = result.boxes
            for xyxy, conf, cls in zip(boxes.xyxyn.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                raw = result.names[int(cls)]
                found.append(Detection(
                    label=self._label_map.get(raw, raw), raw_label=raw, conf=float(conf), box=_box(xyxy),
                ))

        if self._pose is not None:
            result = self._pose.predict(
                image, conf=self._conf, imgsz=self._imgsz, device=self._device, verbose=False,
            )[0]
            boxes = result.boxes
            kp_xy = result.keypoints.xyn.tolist() if result.keypoints is not None else []
            kp_conf = (
                result.keypoints.conf.tolist()
                if result.keypoints is not None and result.keypoints.conf is not None else []
            )
            for i, (xyxy, conf) in enumerate(zip(boxes.xyxyn.tolist(), boxes.conf.tolist())):
                keypoints = None
                if i < len(kp_xy) and i < len(kp_conf):
                    keypoints = {
                        name: (float(x), float(y), float(c))
                        for name, (x, y), c in zip(COCO_KEYPOINTS, kp_xy[i], kp_conf[i])
                    }
                found.append(Detection(
                    label=self._label_map.get("person", "person"), raw_label="person",
                    conf=float(conf), box=_box(xyxy), keypoints=keypoints,
                ))
        return found


def _box(xyxy: list[float]) -> Box:
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in xyxy)
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)
