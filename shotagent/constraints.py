"""Relation predicates: turn a relation declared in the plan into a verdict on one frame.

A constraint in the plan (plans.py) only says which predicate, on whom, with what parameters. For example

    predicate="apart", args={"a": "person", "b": "bag", "ruler": "table", "min_ratio": 0.5}

reads "the horizontal gap between person and bag, measured in table widths, must exceed 0.5". There are no
coordinates: the bag can be anywhere, and the guidance is worked out afresh for every frame.

Two entry points, both for the fast loop:
    evaluate()  a constraint + this frame's World -> CheckResult (satisfied / not satisfied / cannot tell + hint)
    brief()     a constraint + the scene facts in the ledger -> one line of guidance before the take ("bag on the left -> person on the right")

World is this frame as the fast loop sees it: live detections win; something not visible in this frame
is recovered from the ledger when the ledger holds its position relative to an anchor and that anchor is visible now.
Every verdict notes in basis whether each operand came from the live picture or from the ledger.

The predicates and the placeholders they give to hint templates ({a} {b} {ruler} and numeric arguments are always available):
    in_frame        a is in the frame                              -
    under           a is under b                                   -
    apart           gap between a and b > min_ratio ruler widths   {ratio} {away}
    not_facing      a is not facing b                              {toward} {facing}
    pitch_at_least  camera pitch >= min_deg (needs a sensor)       {deg}

To add a relation: write a check function and register it in PREDICATES with its spec (arguments, checker, default wording).
Hand-written plans can use it at once, and generated lists get it in the menu shown to the model automatically (plan_gen.py).

Dependencies: contracts, config, spatial, texts, ledger.state (only for the SceneFact type).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from . import texts
from .config import Tuning
from .contracts import (
    CAP_DETECT, CAP_POSE, Box, Checker, CheckResult, CheckState, Constraint, Detection, Observation,
)
from .ledger.state import SceneFact
from .spatial import DEFAULT_TUNING, facing_from_keypoints, gap_ratio, opposite, project, relative_position

LIVE, LEDGER = "live", "ledger"

Names = Mapping[str, str]
Memory = Mapping[str, SceneFact]


# ───────────────────────── This frame's world ─────────────────────────

@dataclass(frozen=True)
class Located:
    label: str
    box: Box
    basis: str                              # LIVE | LEDGER
    detection: Optional[Detection] = None   # set when basis == LIVE
    fact: Optional[SceneFact] = None        # set when basis == LEDGER


class World:
    def __init__(self, observation: Observation, memory: Memory, tuning: Tuning = DEFAULT_TUNING):
        self.observation = observation
        self.memory = memory        # position facts from the ledger snapshot: role name -> SceneFact
        self.tuning = tuning
        self.sensors = observation.sensors

    def live(self, label: str) -> Optional[Detection]:
        """Detected live in this frame (the most confident one when there are several of a kind)."""
        candidates = [d for d in self.observation.detections if d.label == label]
        return max(candidates, key=lambda d: d.conf) if candidates else None

    def locate(self, label: str) -> Optional[Located]:
        """Live first; if not visible, use the ledger and project it into the current picture through its anchor."""
        detection = self.live(label)
        if detection is not None:
            return Located(label, detection.box, LIVE, detection=detection)
        fact = self.memory.get(label)
        if fact is not None:
            anchor = self.live(str(fact.value.get("anchor", "")))
            if anchor is not None:
                return Located(label, project(fact.value, anchor.box), LEDGER, fact=fact)
        return None


# ───────────────────────── Predicates ─────────────────────────

@dataclass(frozen=True)
class Outcome:
    state: CheckState
    slots: dict[str, Any] = field(default_factory=dict)       # placeholders for the hint template
    measured: dict[str, Any] = field(default_factory=dict)    # the measured numbers
    readout: str = ""                                         # the same result as a short phrase for people
    basis: dict[str, str] = field(default_factory=dict)
    unknown: str = ""                                         # explanation when it cannot tell


def _unknown(message: str) -> Outcome:
    return Outcome(CheckState.UNKNOWN, unknown=message)


def _in_frame(args: Mapping[str, Any], world: World, names: Names) -> Outcome:
    a = args["a"]
    detection = world.live(a)   # "in frame" only counts the live picture, never the ledger
    if detection is None:
        return Outcome(CheckState.VIOLATED, basis={a: LIVE})
    return Outcome(CheckState.OK, measured={"conf": round(detection.conf, 2)}, basis={a: LIVE})


def _under(args: Mapping[str, Any], world: World, names: Names) -> Outcome:
    a, b = args["a"], args["b"]
    da, db = world.live(a), world.live(b)
    missing = [label for label, det in ((a, da), (b, db)) if det is None]
    if missing:
        return _unknown(texts.not_in_frame(missing, names))
    rel = relative_position(da.box, db.box, world.tuning)
    if rel is None:
        return _unknown(texts.not_in_frame([b], names))
    state = CheckState.OK if rel.vertical == "under" else CheckState.VIOLATED
    return Outcome(
        state,
        measured={"u": round(rel.u, 2), "v": round(rel.v, 2), "vertical": rel.vertical},
        basis={a: LIVE, b: LIVE},
    )


def _apart(args: Mapping[str, Any], world: World, names: Names) -> Outcome:
    a, b, ruler = args["a"], args["b"], args["ruler"]
    min_ratio = float(args.get("min_ratio", 0.5))

    da = world.live(a)
    if da is None:
        return _unknown(texts.not_in_frame([a], names))
    dr = world.live(ruler)
    if dr is None:
        return _unknown(texts.ruler_missing(ruler, names))
    lb = world.locate(b)   # b may come from the ledger: e.g. the bag is hidden, but the ledger remembers it under the table on the left
    if lb is None:
        return _unknown(texts.no_memory(b, names))

    ratio = gap_ratio(da.box, lb.box, dr.box)
    rel_b = relative_position(lb.box, dr.box, world.tuning)
    if ratio is None or rel_b is None:
        return _unknown(texts.ruler_missing(ruler, names))

    # Which way to move: a goes to the side of the ruler opposite b; with b dead centre, a keeps going the way it already leans
    if rel_b.side in ("left", "right"):
        away = opposite(rel_b.side)
    else:
        away = "right" if da.box.cx >= lb.box.cx else "left"

    return Outcome(
        CheckState.OK if ratio > min_ratio else CheckState.VIOLATED,
        slots={"ratio": f"{ratio:.2f}", "min_ratio": f"{min_ratio:g}", "away": texts.SIDE[away]},
        measured={"ratio": round(ratio, 3), "min_ratio": min_ratio, "away": away},
        readout=f"{ratio:.2f} {texts.name_of(ruler, names)} widths",
        basis={a: LIVE, ruler: LIVE, b: lb.basis},
    )


def _not_facing(args: Mapping[str, Any], world: World, names: Names) -> Outcome:
    a, b = args["a"], args["b"]
    da = world.live(a)
    if da is None:
        return _unknown(texts.not_in_frame([a], names))
    lb = world.locate(b)
    if lb is None:
        return _unknown(texts.no_memory(b, names))
    facing = facing_from_keypoints(da.keypoints, world.tuning)
    if facing is None:
        return _unknown(texts.cannot_tell_facing(a, names))

    dx = lb.box.cx - da.box.cx
    toward = "left" if dx < 0 else "right"
    overlapping = abs(dx) < 0.5 * da.box.w   # b is right in front of a / at their feet: "facing which way" does not apply
    violated = facing == toward and not overlapping
    return Outcome(
        CheckState.VIOLATED if violated else CheckState.OK,
        slots={"toward": texts.SIDE[toward], "facing": texts.FACING[facing]},
        measured={"facing": facing, "toward": toward},
        readout=texts.FACING[facing],
        basis={a: LIVE, b: lb.basis},
    )


def _pitch_at_least(args: Mapping[str, Any], world: World, names: Names) -> Outcome:
    min_deg = float(args.get("min_deg", 0))
    pitch = world.sensors.get("pitch")
    if pitch is None:
        return _unknown(texts.NO_SENSOR_READING)
    return Outcome(
        CheckState.OK if pitch >= min_deg else CheckState.VIOLATED,
        slots={"deg": f"{pitch:.0f}", "min_deg": f"{min_deg:g}"},
        measured={"deg": round(float(pitch), 1), "min_deg": min_deg},
        readout=f"{pitch:.0f}°",
        basis={"camera": "sensor"},
    )


# -- Pre-take guidance derived from the ledger --

def _origin(fact: SceneFact, names: Names) -> str:
    source = texts.SOURCE_NAME.get(fact.source.value, fact.source.value)
    return f"The ledger has the {texts.name_of(fact.subject, names)} {fact.label} ({source}, {fact.take_id})"


# The ledger stores "which side of the anchor", so this guidance is also phrased relative to the anchor
# ("right side of the table"), not the frame: it stays true when the framing changes. The per-frame fix hints are the ones that speak of frame left / right.

def _brief_apart(args: Mapping[str, Any], memory: Memory, names: Names) -> Optional[str]:
    fact = memory.get(args["b"])
    if fact is None or fact.value.get("anchor") != args["ruler"]:
        return None
    a, ruler = texts.name_of(args["a"], names), texts.name_of(args["ruler"], names)
    side = fact.value.get("side")
    if side in ("left", "right"):
        return f"{_origin(fact, names)}, so keep the {a} on the {texts.SIDE_OF_ANCHOR[opposite(side)]} of the {ruler}"
    return f"{_origin(fact, names)}; left or right is unclear, so keep the {a} as far away as you can"


def _brief_not_facing(args: Mapping[str, Any], memory: Memory, names: Names) -> Optional[str]:
    fact = memory.get(args["b"])
    side = fact.value.get("side") if fact is not None else None
    if side not in ("left", "right"):
        return None
    a, b = texts.name_of(args["a"], names), texts.name_of(args["b"], names)
    anchor = texts.name_of(str(fact.value.get("anchor", "")), names)
    return f"The {b} is on the {texts.SIDE_OF_ANCHOR[side]} of the {anchor}: the {a} should not look that way"


@dataclass(frozen=True)
class Predicate:
    check: Callable[[Mapping[str, Any], World, Names], Outcome]
    brief: Optional[Callable[[Mapping[str, Any], Memory, Names], Optional[str]]] = None

    # -- Spec: what arguments this predicate takes, who checks it, its default wording --
    # When a shot list is generated, both the menu shown to the model and the validation of its output come from here (plan_gen.py),
    # so a newly registered predicate is usable in generated lists at once, with no prompt to edit.
    checker: Checker = Checker.YOLO
    roles: tuple[str, ...] = ()                    # which arguments are role names
    numbers: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)   # numeric argument -> (default, min, max)
    only: Mapping[str, tuple[str, ...]] = field(default_factory=dict)   # a role argument restricted to these roles
    needs: tuple[str, ...] = ()                    # sensors required when checker=SENSOR
    meaning: str = ""                              # one-line explanation (shown to the model)
    text: str = ""                                 # template for the constraint's description
    fix: str = ""                                  # hint template when not satisfied
    prompt: str = ""                               # template when degraded to a text prompt


PREDICATES: dict[str, Predicate] = {
    "in_frame": Predicate(
        _in_frame, roles=("a",),
        meaning="a is visible in the frame",
        text="The {a} is in frame", fix="Get the {a} in frame: there is no {a} in the picture",
    ),
    "under": Predicate(
        _under, roles=("a", "b"),
        meaning="a is underneath b (b is furniture such as a table, chair or bed); both must be in the frame",
        text="The {a} is under the {b}", fix="Put the {a} under the {b}, or move the camera so the space under the {b} is in frame",
    ),
    "apart": Predicate(
        _apart, _brief_apart, roles=("a", "b", "ruler"), numbers={"min_ratio": (0.5, 0.1, 3.0)},
        meaning=("the horizontal gap between a and b is larger than min_ratio times the width of ruler; "
                 "ruler is a third subject used as the yardstick (usually the furniture) and must be in the frame; "
                 "b may be hidden if an earlier shot remembered where it sits relative to ruler"),
        text="The {a} and the {b} are more than {min_ratio} {ruler} widths apart",
        fix="Move the {a} toward {away}: too close to the {b} ({ratio} {ruler} widths now, needs more than {min_ratio})",
    ),
    "not_facing": Predicate(
        _not_facing, _brief_not_facing, checker=Checker.POSE, roles=("a", "b"), only={"a": ("person",)},
        meaning=("the person a is not turned toward the side where b is (coarse: left / right / facing camera); "
                 "b may be hidden if an earlier shot remembered its position"),
        text="The {a} is not facing the {b}",
        fix="Have the {a} turn the other way or face the camera: they are facing the {b}'s side ({toward})",
        prompt="Confirm the {a} is not looking toward the {b} (no pose estimation right now, cannot be checked automatically)",
    ),
    "pitch_at_least": Predicate(
        _pitch_at_least, checker=Checker.SENSOR, needs=("gyro",), numbers={"min_deg": (25, 5, 80)},
        meaning=("the camera is tilted down by at least min_deg degrees (a high-angle shot); needs a gyro, "
                 "so on a plain webcam it falls back to a text reminder"),
        text="Camera pitch ≥ {min_deg}°",
        fix="Raise the camera and tilt it further down: pitch is only {deg}°, needs ≥ {min_deg}°",
        prompt="Shoot down from a high position (this video source has no gyro, so the pitch cannot be checked automatically: confirm it yourself)",
    ),
}


# ───────────────────────── Entry points ─────────────────────────

class _KeepUnknown(dict):
    """Leaves placeholders the template uses but nobody supplied as they are, so hint wording cannot crash the fast loop."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _slots(constraint: Constraint, names: Names, extra: Optional[Mapping[str, Any]] = None) -> _KeepUnknown:
    slots = _KeepUnknown()
    for key, value in constraint.args.items():
        slots[key] = texts.name_of(value, names) if isinstance(value, str) else value
    slots.update(extra or {})
    return slots


def evaluate(constraint: Constraint, world: World, names: Names) -> CheckResult:
    """Check one fast constraint against one frame."""
    predicate = PREDICATES.get(constraint.predicate)
    if predicate is None:
        return CheckResult(
            constraint_id=constraint.id, state=CheckState.UNKNOWN,
            message=f"The plan uses a predicate that is not registered: {constraint.predicate!r}",
        )
    outcome = predicate.check(constraint.args, world, names)
    if outcome.state is CheckState.OK:
        message = ""
    elif outcome.state is CheckState.VIOLATED:
        message = (constraint.fix or f"Not satisfied yet: {constraint.text}").format_map(
            _slots(constraint, names, outcome.slots))
    else:
        message = outcome.unknown or f"Can't tell yet: {constraint.text}"
    return CheckResult(
        constraint_id=constraint.id, state=outcome.state, message=message,
        readout=outcome.readout, measured=outcome.measured, basis=outcome.basis,
    )


def brief(constraint: Constraint, memory: Memory, names: Names) -> Optional[str]:
    """Whether this constraint yields a line of pre-take guidance from the scene facts in the ledger."""
    predicate = PREDICATES.get(constraint.predicate)
    if predicate is None or predicate.brief is None:
        return None
    return predicate.brief(constraint.args, memory, names)


def required_caps(constraint: Constraint) -> frozenset[str]:
    """Capabilities needed to check this constraint automatically (the detector's + the video source's sensors)."""
    if constraint.checker is Checker.YOLO:
        return frozenset({CAP_DETECT})
    if constraint.checker is Checker.POSE:
        return frozenset({CAP_DETECT, CAP_POSE})
    if constraint.checker is Checker.SENSOR:
        return frozenset(constraint.needs)
    return frozenset()


def degraded_prompt(constraint: Constraint, names: Names) -> str:
    """The text prompt a constraint degrades to when it cannot be checked automatically."""
    template = constraint.prompt or f"Please confirm: {constraint.text}{texts.CANNOT_AUTO_CHECK}"
    return template.format_map(_slots(constraint, names))


def subjects_of(constraint: Constraint) -> set[str]:
    """Which roles this constraint mentions."""
    return {v for k, v in constraint.args.items() if k in ("a", "b", "ruler") and isinstance(v, str)}
