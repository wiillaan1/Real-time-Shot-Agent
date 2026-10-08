"""Simulated picture: runs the whole loop with no camera and no model.

    render()             draws a schematic picture from a SimScene (table, bag, person; the colours are an agreed code)
    SyntheticDetector    decodes the table, the bag, the person and the person's facing from that picture's pixels

Drawing and recognising sit in one file and share one palette and one geometry, so the two always agree.
It is still "image in, detections out": the pipeline is the same one a real camera goes through,
only YOLO is replaced by finding blobs of colour.

Dependencies: contracts only.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from ..contracts import CAP_DETECT, CAP_POSE, Box, Detection, Frozen

WIDTH, HEIGHT = 640, 360

# Palette (BGR). Any two colours differ by more than 60 in at least one channel, so they survive JPEG compression.
BACKGROUND = (236, 236, 236)
FLOOR = (205, 205, 205)
TABLE = (50, 100, 150)
BAG = (40, 40, 220)
BODY = (200, 80, 40)
HEAD = (160, 200, 240)
NOSE = (0, 210, 250)
_TOLERANCE = 30

# Scene geometry (normalised coordinates)
TABLE_X1, TABLE_X2 = 0.25, 0.75
TABLE_TOP_Y1, TABLE_TOP_Y2 = 0.52, 0.57
FLOOR_Y = 0.92
BAG_W, BAG_H = 0.09, 0.12
BODY_HALF_W, BODY_Y1, BODY_Y2 = 0.06, 0.30, 0.75
HEAD_Y, HEAD_R = 0.22, 0.07            # head radius is relative to the picture height
NOSE_OFFSET, NOSE_R = 0.8, 0.22        # both in multiples of the head radius


class SimScene(Frozen):
    """What is in the simulated picture. The sliders in the UI change these fields."""

    table: bool = True                 # whether the table is in the picture
    bag: Optional[float] = 0.2         # the bag's centre along the table's width (0 = left edge, 1 = right edge); None = no bag
    bag_under: bool = True             # True under the table, False on top of it
    person: Optional[float] = None     # the person's centre along the picture's width (0..1); None = no person
    facing: str = "front"              # left | right | front
    pitch: float = 0.0                 # camera pitch in degrees. The picture does not change with it; it is only sent along as the gyro reading


def render(scene: SimScene, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    def px(x: float, y: float) -> tuple[int, int]:
        return int(round(x * width)), int(round(y * height))

    image = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    cv2.rectangle(image, px(0, FLOOR_Y), px(1, 1), FLOOR, -1)

    # The person is drawn first (seated behind the table, partly hidden by it)
    if scene.person is not None:
        x = scene.person
        cv2.rectangle(image, px(x - BODY_HALF_W, BODY_Y1), px(x + BODY_HALF_W, BODY_Y2), BODY, -1)
        center = px(x, HEAD_Y)
        radius = int(round(HEAD_R * height))
        cv2.circle(image, center, radius, HEAD, -1)
        shift = {"left": -1, "right": 1}.get(scene.facing, 0) * NOSE_OFFSET * radius
        cv2.circle(image, (int(round(center[0] + shift)), center[1]), int(round(NOSE_R * radius)), NOSE, -1)

    if scene.table:
        cv2.rectangle(image, px(TABLE_X1, TABLE_TOP_Y1), px(TABLE_X2, TABLE_TOP_Y2), TABLE, -1)
        for leg_x in (TABLE_X1 + 0.02, TABLE_X2 - 0.04):
            cv2.rectangle(image, px(leg_x, TABLE_TOP_Y2), px(leg_x + 0.02, FLOOR_Y), TABLE, -1)

    if scene.bag is not None:
        cx = TABLE_X1 + scene.bag * (TABLE_X2 - TABLE_X1)
        y2 = FLOOR_Y if scene.bag_under else TABLE_TOP_Y1
        cv2.rectangle(image, px(cx - BAG_W / 2, y2 - BAG_H), px(cx + BAG_W / 2, y2), BAG, -1)

    caption = f"SIM  pitch {scene.pitch:.0f} deg"
    cv2.putText(image, caption, (10, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    return image


class SyntheticDetector:
    """The simulated picture's "detector": finds blobs of the agreed colours. Same capabilities as real YOLO + a pose model."""

    name = "synthetic"
    capabilities = frozenset({CAP_DETECT, CAP_POSE})
    note = "Simulated picture: decoded by colour, not a real model"
    roles = frozenset({"person", "bag", "table"})   # the simulated picture can only draw these three

    def detect(self, image: Any) -> list[Detection]:
        height, width = image.shape[:2]
        found: list[Detection] = []

        table = _blob_box(image, TABLE, width, height)
        if table is not None:
            found.append(Detection(label="table", raw_label="sim-table", box=table))

        bag = _blob_box(image, BAG, width, height)
        if bag is not None:
            found.append(Detection(label="bag", raw_label="sim-bag", box=bag))

        body = _blob_box(image, BODY, width, height)
        head = _blob_box(image, HEAD, width, height)
        if body is not None and head is not None:
            box = Box(
                x1=min(body.x1, head.x1), y1=min(body.y1, head.y1),
                x2=max(body.x2, head.x2), y2=max(body.y2, head.y2),
            )
            found.append(Detection(
                label="person", raw_label="sim-person", box=box,
                keypoints=_keypoints(body, head, _blob_box(image, NOSE, width, height, min_area=4)),
            ))
        return found


def _blob_box(image: np.ndarray, color: tuple[int, int, int], width: int, height: int,
              min_area: Optional[int] = None) -> Optional[Box]:
    """Bounding box around all large blobs of one colour. Small specks (compression noise) do not count."""
    lower = np.clip(np.array(color) - _TOLERANCE, 0, 255).astype(np.uint8)
    upper = np.clip(np.array(color) + _TOLERANCE, 0, 255).astype(np.uint8)
    mask = cv2.inRange(image, lower, upper)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    threshold = min_area if min_area is not None else max(int(0.0004 * width * height), 12)
    boxes = [
        (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
         stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH],
         stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= threshold
    ]
    if not boxes:
        return None
    return Box(
        x1=min(b[0] for b in boxes) / width, y1=min(b[1] for b in boxes) / height,
        x2=max(b[2] for b in boxes) / width, y2=max(b[3] for b in boxes) / height,
    )


def _keypoints(body: Box, head: Box, nose: Optional[Box]) -> dict[str, tuple[float, float, float]]:
    """Turn the blobs into keypoints named like the YOLO pose model's, so the facing logic downstream is the same code."""
    points: dict[str, tuple[float, float, float]] = {
        "left_shoulder": (body.x1, body.y1, 1.0),
        "right_shoulder": (body.x2, body.y1, 1.0),
    }
    if nose is None:
        return points
    points["nose"] = (nose.cx, nose.cy, 1.0)
    turned = abs(nose.cx - head.cx) > 0.25 * (head.w / 2)
    if turned:
        # Turned sideways: only the ear on the camera side is visible, in the middle of the head
        points["left_ear"] = (head.cx, head.cy, 1.0)
    else:
        points["left_ear"] = (head.cx - 0.45 * head.w, head.cy, 1.0)
        points["right_ear"] = (head.cx + 0.45 * head.w, head.cy, 1.0)
    return points
