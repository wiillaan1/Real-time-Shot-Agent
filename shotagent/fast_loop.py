"""Fast loop: one frame in, one framing hint out (well under a second).

    load(snapshot, shot_id, caps)   called once when a shot starts: reads the ledger snapshot and the source's capabilities into memory
    step(frame) -> Guidance         called for every frame: detect -> check each constraint -> hint

"Reads a ledger snapshot, never writes the ledger" is guaranteed by structure, not by discipline: this file does not
import ledger.store and FastLoop holds no Ledger object. Its only view of the ledger is the immutable snapshot passed to load().
What it detects (Observation) travels out with the Guidance, is collected into the take by the session,
and is finally written to the ledger by the slow loop. So YOLO's positions reach the ledger without the fast loop touching it.

load() settles three things up front:
  1. which detector to use (by source type: the decoder for the simulated picture, YOLO or the null detector for a camera)
  2. how each constraint is handled:
       check     checked automatically
       degraded  a needed capability (detection / pose / sensor) is missing -> reduced to a text prompt that says what is missing
       slow      slow-loop constraint, judged after the take
  3. pre-take guidance: which constraints yield a line straight from the scene facts in the ledger ("bag on the left -> person on the right")

Dependencies: contracts, config, constraints, spatial, texts, perception.base, ledger.state.
Never: ledger.store, ledger.events, slow_loop, judge, takes, session.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from . import texts
from .config import Tuning
from .constraints import World, brief, degraded_prompt, evaluate, required_caps, subjects_of
from .contracts import (
    Checker, CheckResult, CheckState, Frame, Guidance, Loop, Observation, Recalled, Shot, SourceCaps,
)
from .ledger.state import LedgerState, SceneFact
from .perception.base import Detector
from .spatial import relative_position

MODE_CHECK, MODE_DEGRADED, MODE_SLOW = "check", "degraded", "slow"


@dataclass(frozen=True)
class ConstraintMode:
    mode: str           # MODE_CHECK | MODE_DEGRADED | MODE_SLOW
    note: str = ""      # why this mode (when degraded: which capability is missing)
    prompt: str = ""    # the text prompt shown to the user once degraded


@dataclass(frozen=True)
class _Context:
    """What stays fixed for the length of a shot. step() takes one reference at the start, so a load() mid-step does not affect it."""

    detector: Detector
    capabilities: frozenset[str]
    shot: Optional[Shot] = None
    names: Mapping[str, str] = field(default_factory=dict)
    memory: Mapping[str, SceneFact] = field(default_factory=dict)   # position facts from the ledger
    modes: Mapping[str, ConstraintMode] = field(default_factory=dict)
    briefing: tuple[str, ...] = ()
    watch: frozenset[str] = frozenset()   # roles to watch for "picture disagrees with ledger"
    recallable: frozenset[str] = frozenset()   # roles this shot uses whose position the ledger holds
    ledger_version: int = 0


class FastLoop:
    def __init__(self, pick_detector: Callable[[Optional[SourceCaps]], Detector], tuning: Tuning):
        self._pick_detector = pick_detector
        self._tuning = tuning
        self._ctx = _Context(detector=pick_detector(None), capabilities=frozenset())

    # ───────────── When a shot starts: read the snapshot ─────────────

    def load(self, snapshot: LedgerState, shot_id: Optional[str], caps: Optional[SourceCaps]) -> None:
        detector = self._pick_detector(caps)
        capabilities = frozenset(detector.capabilities) | frozenset(caps.sensors if caps else ())

        record = snapshot.record(shot_id) if shot_id else None
        shot = record.shot if record else None
        names = dict(snapshot.subjects)
        memory = {f.subject: f for f in snapshot.scene if f.field == "position"}

        modes: dict[str, ConstraintMode] = {}
        briefing: list[str] = []
        mentioned: set[str] = set()      # roles mentioned by automatically checkable constraints
        established: set[str] = set()    # roles whose position this shot itself (re)establishes
        if shot is not None:
            for c in shot.constraints:
                if c.loop is Loop.SLOW:
                    modes[c.id] = ConstraintMode(MODE_SLOW, texts.SLOW_PENDING)
                    continue
                if c.checker is Checker.NONE:
                    modes[c.id] = ConstraintMode(MODE_DEGRADED, texts.TEXT_ONLY, degraded_prompt(c, names))
                else:
                    lacking = required_caps(c) - capabilities
                    if lacking:
                        modes[c.id] = ConstraintMode(
                            MODE_DEGRADED, texts.missing_caps(lacking), degraded_prompt(c, names))
                    else:
                        modes[c.id] = ConstraintMode(MODE_CHECK)
                        mentioned |= subjects_of(c)
                # Guidance derived from the ledger does not need a detector: it is given even when the constraint is degraded
                line = brief(c, memory, names)
                if line:
                    briefing.append(line)
            established = {e.subject for e in shot.establishes}

        remembered = mentioned & set(memory)
        self._ctx = _Context(
            detector=detector, capabilities=capabilities, shot=shot, names=names, memory=memory,
            modes=modes, briefing=tuple(briefing),
            # A position being re-established is expected to differ from the ledger: no need to flag it
            watch=frozenset(remembered - established),
            recallable=frozenset(remembered), ledger_version=snapshot.version,
        )

    # ───────────── Every frame ─────────────

    def step(self, frame: Frame) -> Guidance:
        ctx = self._ctx
        started = time.perf_counter()

        detections = ctx.detector.detect(frame.image)
        if ctx.names:
            # The detector knows more things than this list uses (chairs, cups...): what the list does not mention is dropped here, so no extra boxes on screen
            detections = [d for d in detections if d.label in ctx.names]
        observation = Observation(
            seq=frame.seq, ts=frame.ts, detections=tuple(detections), sensors=dict(frame.sensors),
        )

        def done(**fields: Any) -> Guidance:
            return Guidance(
                seq=frame.seq, shot_id=ctx.shot.id if ctx.shot else None, observation=observation,
                latency_ms=round((time.perf_counter() - started) * 1000, 1), **fields,
            )

        if ctx.shot is None:
            return done(headline=texts.NO_SHOT, level="info", ready=False)

        world = World(observation, ctx.memory, tuning=self._tuning)

        checks: list[CheckResult] = []
        for c in ctx.shot.constraints:
            mode = ctx.modes[c.id]
            if mode.mode == MODE_SLOW:
                checks.append(CheckResult(constraint_id=c.id, state=CheckState.PENDING, message=mode.note))
            elif mode.mode == MODE_DEGRADED:
                checks.append(CheckResult(constraint_id=c.id, state=CheckState.DEGRADED, message=mode.prompt))
            else:
                checks.append(evaluate(c, world, ctx.names))

        headline, level, ready = _headline(checks)
        return done(
            headline=headline, level=level, ready=ready, checks=tuple(checks),
            notes=self._conflicts(world, ctx), recalled=self._recalled(world, ctx),
        )

    @staticmethod
    def _recalled(world: World, ctx: _Context) -> tuple[Recalled, ...]:
        """Which subjects in this frame were recovered from the ledger (not visible, but their anchor is)."""
        found = []
        for label in sorted(ctx.recallable):
            located = world.locate(label)
            if located is not None and located.fact is not None:
                found.append(Recalled(label=label, box=located.box, take_id=located.fact.take_id))
        return tuple(found)

    def _conflicts(self, world: World, ctx: _Context) -> tuple[str, ...]:
        """Say so when the picture and the ledger disagree on left / right (the minimal conflict handling: trust the picture, tell the user)."""
        notes = []
        for label in sorted(ctx.watch):
            fact = ctx.memory[label]
            anchor_label = str(fact.value.get("anchor", ""))
            seen, anchor = world.live(label), world.live(anchor_label)
            if seen is None or anchor is None:
                continue
            rel = relative_position(seen.box, anchor.box, self._tuning)
            if rel is None:
                continue
            if {rel.side, fact.value.get("side")} == {"left", "right"}:
                now = texts.position_label(texts.name_of(anchor_label, ctx.names), rel.vertical, rel.side)
                notes.append(
                    f"The picture shows the {texts.name_of(label, ctx.names)} {now}, but the ledger says {fact.label} ({fact.take_id}). "
                    "Hints follow the picture; the ledger only changes when that shot is reshot"
                )
        return tuple(notes)

    # ───────────── For the session / the UI ─────────────

    def describe(self) -> dict[str, Any]:
        ctx = self._ctx
        roles = getattr(ctx.detector, "roles", None)
        return {
            "detector": ctx.detector.name,
            "detector_note": ctx.detector.note,
            "capabilities": sorted(ctx.capabilities),
            "roles": sorted(roles) if roles is not None else None,   # roles this detector recognises; None = no limit
            "shot_id": ctx.shot.id if ctx.shot else None,
            "briefing": list(ctx.briefing),
            "modes": {cid: {"mode": m.mode, "note": m.note} for cid, m in ctx.modes.items()},
            "ledger_version": ctx.ledger_version,
        }


def _headline(checks: list[CheckResult]) -> tuple[str, str, bool]:
    """Pick the one line to say first. The order of constraints in the plan is their priority.

    Degraded constraints (no automatic check, text prompt only) do not block the take, but they do not count as satisfied either:
    when there are any, the verdict says some items are left for the user to confirm instead of claiming everything passes.
    Their text prompts are in checks (the message of state=DEGRADED entries); the UI lists them under the hint.
    """
    for check in checks:
        if check.state in (CheckState.VIOLATED, CheckState.UNKNOWN):
            return check.message, "fix", False
    self_check = any(check.state is CheckState.DEGRADED for check in checks)
    if any(check.state is CheckState.OK for check in checks):
        return (texts.READY_SELF_CHECK if self_check else texts.READY), "ok", True
    return (texts.SELF_CHECK_ONLY if self_check else texts.READY_NO_CHECKS), "info", True
