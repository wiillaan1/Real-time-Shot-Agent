"""Geometry for relative position and facing (pure functions, no state).

Design doc 2.3: constraints store relations, not positions, and distance uses a relative scale (the table is the ruler), not pixels.
This file is that ruler:

    relative_position()  two boxes -> the subject's position relative to the anchor (u, v, left/right, under/above)   <- the slow loop uses it to write YOLO observations as scene facts
    project()            a scene fact + the anchor as seen now -> roughly where the subject is now                  <- the fast loop uses it to recover hidden subjects from the ledger
    facing_from_keypoints()  pose keypoints -> coarse facing (left / right / front / back)

Encoding and decoding sit in one file so what the slow loop writes to the ledger and what the fast loop reads back follow the same convention.

Dependencies: contracts, config.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Optional, Sequence

from .config import Tuning
from .contracts import Box

DEFAULT_TUNING = Tuning()


@dataclass(frozen=True)
class RelPos:
    """The subject's position relative to the anchor. All ratios, independent of resolution and distance."""

    u: float        # horizontal position of the subject's centre within the anchor box: 0 = left edge, 1 = right edge
    v: float        # vertical position: 0 = top edge, 1 = bottom edge (may fall below 0 or above 1)
    sw: float       # subject width / anchor width
    sh: float       # subject height / anchor height
    side: str       # left | right | center
    vertical: str   # under | above | beside

    def as_value(self, anchor: str) -> dict[str, Any]:
        """The form written into the ledger's SceneFact.value."""
        return {
            "anchor": anchor,
            "u": round(self.u, 3),
            "v": round(self.v, 3),
            "sw": round(self.sw, 3),
            "sh": round(self.sh, 3),
            "side": self.side,
            "vertical": self.vertical,
        }


def side_of(u: float, tuning: Tuning = DEFAULT_TUNING) -> str:
    if u < 0.5 - tuning.center_band:
        return "left"
    if u > 0.5 + tuning.center_band:
        return "right"
    return "center"


def opposite(side: str) -> str:
    return {"left": "right", "right": "left"}.get(side, side)


def _vertical_of(u: float, v: float, tuning: Tuning) -> str:
    within = -tuning.under_margin_u <= u <= 1 + tuning.under_margin_u
    if within and v >= tuning.under_min_v:
        return "under"
    if within and v <= tuning.above_max_v:
        return "above"
    return "beside"


def relative_position(subject: Box, anchor: Box, tuning: Tuning = DEFAULT_TUNING) -> Optional[RelPos]:
    """Position of the subject box relative to the anchor box. None when the anchor box is degenerate (zero width or height)."""
    if anchor.w <= 1e-6 or anchor.h <= 1e-6:
        return None
    u = (subject.cx - anchor.x1) / anchor.w
    v = (subject.cy - anchor.y1) / anchor.h
    return RelPos(
        u=u,
        v=v,
        sw=subject.w / anchor.w,
        sh=subject.h / anchor.h,
        side=side_of(u, tuning),
        vertical=_vertical_of(u, v, tuning),
    )


def aggregate(samples: Sequence[RelPos], tuning: Tuning = DEFAULT_TUNING) -> Optional[RelPos]:
    """Merge a take's per-frame relative positions into one: the median of each component, to resist jitter."""
    if not samples:
        return None
    u = median(s.u for s in samples)
    v = median(s.v for s in samples)
    return RelPos(
        u=u,
        v=v,
        sw=median(s.sw for s in samples),
        sh=median(s.sh for s in samples),
        side=side_of(u, tuning),
        vertical=_vertical_of(u, v, tuning),
    )


def project(value: Mapping[str, Any], anchor: Box) -> Box:
    """Inverse of relative_position: the relative position in the ledger + the anchor as seen now -> the subject's box now."""
    cx = anchor.x1 + float(value["u"]) * anchor.w
    cy = anchor.y1 + float(value["v"]) * anchor.h
    w = float(value.get("sw", 0.1)) * anchor.w
    h = float(value.get("sh", 0.1)) * anchor.h
    return Box(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2)


def gap_ratio(a: Box, b: Box, ruler: Box) -> Optional[float]:
    """Horizontal gap between a and b, in units of the ruler's width.

    Horizontal only: with the person seated at the table and the bag under it, their height difference does not mean "far apart".
    """
    if ruler.w <= 1e-6:
        return None
    return abs(a.cx - b.cx) / ruler.w


def facing_from_keypoints(
    keypoints: Optional[Mapping[str, Sequence[float]]],
    tuning: Tuning = DEFAULT_TUNING,
) -> Optional[str]:
    """Coarse facing from pose keypoints: left | right | front | back; None when it cannot tell.

    It looks at how far the nose is off the head's midline. The midline comes from a left/right symmetric pair, tried in order of reliability:
      1. Both ears clearly visible -> use the ears. As the head turns the nose moves toward one ear: the most sensitive cue.
      2. Otherwise both eyes clearly visible -> use the eyes. Covers hair over the ears or an unclear far ear.
      3. No usable pair (profile) -> which side of the clearest ear (or eye, failing that) the nose is on.
    The offset is divided by half the pair's distance: 0 = nose dead centre, 1 = nose directly under one of the points.
    Everything is a ratio of horizontal distances, so frame size, resolution and distance from the camera do not matter.
    No nose but both ears visible -> back to the camera.

    The design doc calls this "coarse": only a clear turn counts as facing a side, a slight turn still counts as facing the camera.
    The thresholds were set against the YOLO pose model's output on a batch of real photos (Tuning.facing_*); revisit them if the model changes.
    """
    if not keypoints:
        return None

    def x_of(name: str, min_conf: float) -> Optional[float]:
        point = keypoints.get(name)
        if point is None or len(point) < 3 or point[2] < min_conf:
            return None
        return float(point[0])

    nose = x_of("nose", tuning.kp_min_conf)
    if nose is None:
        ears = [x_of("left_ear", tuning.kp_min_conf), x_of("right_ear", tuning.kp_min_conf)]
        return "back" if None not in ears else None

    for part, threshold in (("ear", tuning.facing_ears_ratio), ("eye", tuning.facing_eyes_ratio)):
        left, right = x_of(f"left_{part}", tuning.kp_pair_conf), x_of(f"right_{part}", tuning.kp_pair_conf)
        if left is None or right is None:
            continue
        offset = nose - (left + right) / 2
        if abs(offset) < threshold * abs(left - right) / 2:
            return "front"
        return "left" if offset < 0 else "right"

    # No trustworthy pair: the head is well turned and the far-side points are the model's guess. Trust only the clearest one (the near ear, or the eye failing that)
    for part in ("ear", "eye"):
        seen = [
            (point[2], float(point[0]))
            for point in (keypoints.get(f"left_{part}"), keypoints.get(f"right_{part}"))
            if point is not None and len(point) >= 3 and point[2] >= tuning.kp_min_conf
        ]
        if seen:
            return "left" if nose < max(seen)[1] else "right"
    return None
