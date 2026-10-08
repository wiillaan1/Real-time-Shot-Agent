"""Shared user-facing wording: names for directions, what to say when something is missing,
the fast loop's verdicts, and the banner for each phase.

Wording lives in three places, decided by who knows best how the sentence should read:
    plans.py     anything tied to a specific shot: its description, each constraint's text, the fix hints
    texts.py     words and sentences shared across shots and modules (this file)
    the module that says it   sentences only one place ever says stay in that place:
                 constraints.py (guidance derived from the ledger), fast_loop.py (picture vs ledger mismatch),
                 slow_loop.py (per-constraint settlement notes), session.py (take results, refused actions),
                 ledger/reducer.py (write log), judge/mock.py (placeholder descriptions)
So translating the whole interface means going through the files above, plus web/index.html and web/js/.

Dependencies: imports no internal module.
"""
from __future__ import annotations

from typing import Iterable, Mapping

# Left and right always mean left and right in the picture (the UI does not mirror the image)
SIDE = {"left": "frame left", "right": "frame right", "center": "the middle of the frame"}
SIDE_OF_ANCHOR = {"left": "left side", "right": "right side", "center": "middle"}
VERTICAL = {"under": "under", "above": "above", "beside": "beside"}
FACING = {"left": "facing frame left", "right": "facing frame right", "front": "facing the camera", "back": "back to the camera"}

SOURCE_NAME = {"plan": "plan", "yolo": "YOLO", "cosmos": "Cosmos", "sensor": "sensor"}


def name_of(label: str, names: Mapping[str, str]) -> str:
    return names.get(label, label)


def join_names(labels: Iterable[str], names: Mapping[str, str]) -> str:
    return " and the ".join(name_of(x, names) for x in labels)


def position_label(anchor_name: str, vertical: str, side: str) -> str:
    """E.g. table + under + left -> "under the table, left side"."""
    where = VERTICAL.get(vertical, "by")
    side_name = SIDE_OF_ANCHOR.get(side, "")
    return f"{where} the {anchor_name}" + (f", {side_name}" if side_name else "")


def not_in_frame(labels: Iterable[str], names: Mapping[str, str]) -> str:
    return f"Can't find the {join_names(labels, names)} in the picture"


def ruler_missing(ruler: str, names: Mapping[str, str]) -> str:
    return f"Can't find the {name_of(ruler, names)} in the picture (it is the yardstick for distance): get it in frame"


def no_memory(label: str, names: Mapping[str, str]) -> str:
    n = name_of(label, names)
    return f"The {n} is not in the picture and the ledger has no position for it yet"


def cannot_tell_facing(label: str, names: Mapping[str, str]) -> str:
    return f"Can't tell which way the {name_of(label, names)} is facing"


_MISSING_CAP = {
    "detect": "no detection model loaded",
    "pose": "no pose estimation",
    "gyro": "the video source has no gyro",
}


def missing_caps(caps: Iterable[str]) -> str:
    """Which capabilities are missing, i.e. why this constraint only gets a text prompt."""
    return ", ".join(_MISSING_CAP.get(c, f"missing {c}") for c in sorted(caps))


TEXT_ONLY = "cannot be checked automatically, text prompt only"
NO_SENSOR_READING = "No sensor reading received"
CANNOT_AUTO_CHECK = " (cannot be checked automatically, please confirm yourself)"
SLOW_PENDING = "judged by the semantic model after the take"

# The fast loop's verdicts. "Self-check" = a requirement that cannot be checked automatically right now
# (it was degraded for lack of a capability); the UI lists its text prompt.
READY = "Framing meets the requirements"
READY_SELF_CHECK = "Everything that can be checked passes. Confirm the items marked \"self-check\" yourself"
SELF_CHECK_ONLY = "Nothing in this shot can be checked automatically right now: go through the \"self-check\" items, then roll"
READY_NO_CHECKS = "This shot has nothing to check before the take: frame it and roll"
NO_SHOT = "No shot left to shoot"

PLAN_CHANGED = "The shot list definition has changed, but the ledger already holds progress, so the old list is still in use. To switch to the new one, click \"Start over\" at the top right (this clears progress)"

BANNER = {
    "planning": "Writing the shot list, one moment",
    "framing": "Framing: adjust as prompted, roll when ready",
    "recording": "Recording",
    "checking": "Checking: stay where you are, move only after it passes",
    "done": "All shots passed",
}
