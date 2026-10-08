"""Slow loop: settle a take once it is shot and write the conclusions to the ledger (a few seconds).

    settle(take) -> Settlement

Settlement has four steps:
  1. Fast constraints: aggregate the per-frame results stored in the take. A constraint passes when the share of
     satisfying frames reaches Tuning.pass_ratio.
     (The fast loop computed these at the time and they were saved with the take. They are not recomputed, so they match what the user saw while shooting.)
  2. Slow constraints: only when the fast ones pass is the take handed to the semantic model (Judge) to answer the yes/no questions and describe it.
     If a fast constraint failed it is not sent: it has to be reshot anyway, and that saves a call of several seconds.
  3. Scene facts (only when the take passes):
       position   where the bag sits relative to the table, aggregated from YOLO's per-frame detections   -> source yolo
       semantic   the semantic model's description of each role                                           -> source cosmos
  4. One atomic commit: TakeSettled + any number of SceneFactObserved.

So "the fast loop never writes the ledger" and "position fields come from YOLO" do not contradict each other:
YOLO's observations are produced by the fast loop and carried in the take; the write happens here, at settlement.

A take passes = every automatically checkable fast constraint passes and every slow constraint passes.
Whether a new take replaces the old one is not decided here: that is a ledger rule (ledger/reducer.py).

Dependencies: contracts, config, constraints, spatial, texts, judge.base,
              ledger.state / ledger.events / ledger.store.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from . import texts
from .config import Tuning
from .constraints import World, subjects_of
from .contracts import (
    Checker, CheckState, Constraint, Establish, JudgeRequest, Loop, SemanticQuestion, Shot,
    ShotStatus, Source, Take,
)
from .judge.base import Judge, JudgeError
from .ledger.events import Event, SceneFactObserved, TakeSettled
from .ledger.state import ConstraintVerdict, LedgerError, LedgerState, SceneFact, TakeRecord
from .ledger.store import Ledger
from .spatial import aggregate, relative_position

_JUDGED = (CheckState.OK, CheckState.VIOLATED, CheckState.UNKNOWN)
_FAST_SOURCE = {
    Checker.YOLO: Source.YOLO,
    Checker.POSE: Source.YOLO,      # pose also comes from a YOLO model
    Checker.SENSOR: Source.SENSOR,
    Checker.NONE: Source.PLAN,      # text-only techniques: there is just the line in the plan, nobody checks it
}


@dataclass(frozen=True)
class Settlement:
    """The outcome of a settlement, handed back to the session to decide what to prompt next."""

    record: TakeRecord
    passed: bool                    # whether this take itself passed
    shot_status: ShotStatus         # the shot's status after settlement
    kept_previous: bool             # this take failed, but the ledger kept the earlier passing one
    reasons: tuple[str, ...]        # why it failed
    facts: tuple[SceneFact, ...]    # facts written to the scene state this time
    state: LedgerState              # ledger snapshot after settlement


class SlowLoop:
    def __init__(self, ledger: Ledger, judge: Judge, tuning: Tuning,
                 clock: Callable[[], float] = time.time):
        self._ledger = ledger
        self._judge = judge
        self._tuning = tuning
        self._clock = clock

    def settle(self, take: Take) -> Settlement:
        state = self._ledger.snapshot()
        record = state.record(take.shot_id)
        if record is None:
            raise LedgerError(f"The ledger has no shot {take.shot_id}")
        shot, names = record.shot, state.subjects

        # 1. Fast constraints: aggregate the per-frame results
        verdicts: dict[str, ConstraintVerdict] = {
            c.id: self._fast_verdict(c, take) for c in shot.constraints if c.loop is Loop.FAST
        }
        fast_ok = all(v.passed is not False for v in verdicts.values())

        # 2. Slow constraints: ask the semantic model
        slow = [c for c in shot.constraints if c.loop is Loop.SLOW]
        description, judge_name, notes = "", "", {}
        if fast_ok:
            verdict = self._judge.judge(self._request(take, shot, slow, names))
            answers = {a.constraint_id: a for a in verdict.answers}
            for c in slow:
                answer = answers.get(c.id)
                if answer is None:
                    raise JudgeError(f"The semantic model did not answer {c.id}")
                verdicts[c.id] = ConstraintVerdict(
                    constraint_id=c.id, text=c.text, passed=answer.passed, source=Source.COSMOS,
                    detail=answer.reason, score=answer.confidence,
                )
            description, judge_name, notes = verdict.description, verdict.model, verdict.subject_notes
        else:
            for c in slow:
                verdicts[c.id] = ConstraintVerdict(
                    constraint_id=c.id, text=c.text, passed=None, source=Source.COSMOS,
                    detail="a fast constraint failed, so this take was not sent to the semantic model",
                )

        ordered = tuple(verdicts[c.id] for c in shot.constraints)
        passed = fast_ok and all(verdicts[c.id].passed is True for c in slow)
        now = self._clock()

        take_record = TakeRecord(
            take_id=take.take_id, shot_id=take.shot_id,
            started_at=take.started_at, ended_at=take.ended_at, n_frames=len(take.frames),
            clip_dir=take.dir, passed=passed, verdicts=ordered,
            description=description, judge=judge_name, settled_at=now,
        )

        # 3. Scene facts: only from a take that passed
        facts: list[SceneFact] = []
        if passed:
            for establish in shot.establishes:
                fact = self._position_fact(establish, take, names, now)
                if fact is not None:
                    facts.append(fact)
            for label, text in notes.items():
                facts.append(SceneFact(
                    subject=label, field="description", value={"text": text}, label=text,
                    source=Source.COSMOS, confidence=1.0, take_id=take.take_id, ts=now,
                ))

        # 4. One atomic commit
        events: list[Event] = [TakeSettled(take=take_record)]
        events += [SceneFactObserved(fact=f) for f in facts]
        new_state = self._ledger.commit(events, writer="slow_loop")

        status = new_state.record(take.shot_id).status
        return Settlement(
            record=take_record, passed=passed, shot_status=status,
            kept_previous=(not passed and status is ShotStatus.PASSED),
            reasons=tuple(f"{v.text} ({v.detail})" if v.detail else v.text
                          for v in ordered if v.passed is False),
            facts=tuple(facts), state=new_state,
        )

    # ───────────── Internals ─────────────

    def _fast_verdict(self, constraint: Constraint, take: Take) -> ConstraintVerdict:
        source = _FAST_SOURCE[constraint.checker]
        states = [
            check.state for frame in take.frames for check in frame.checks
            if check.constraint_id == constraint.id
        ]
        judged = [s for s in states if s in _JUDGED]
        if not judged:
            return ConstraintVerdict(
                constraint_id=constraint.id, text=constraint.text, passed=None, source=source,
                detail="not checked automatically, text prompt only",
            )
        ok = sum(1 for s in judged if s is CheckState.OK)
        ratio = ok / len(judged)
        passed = ratio >= self._tuning.pass_ratio
        detail = f"{ok}/{len(judged)} frames satisfy it"
        if not passed:
            detail += f", below {self._tuning.pass_ratio:.0%}"
        return ConstraintVerdict(
            constraint_id=constraint.id, text=constraint.text, passed=passed, source=source,
            detail=detail, score=round(ratio, 2),
        )

    def _request(self, take: Take, shot: Shot, slow: list[Constraint],
                 names: Mapping[str, str]) -> JudgeRequest:
        mentioned: set[str] = set()
        for c in shot.constraints:
            mentioned |= subjects_of(c)
        for e in shot.establishes:
            mentioned |= {e.subject, e.anchor}
        return JudgeRequest(
            take_id=take.take_id, shot_id=shot.id, shot_title=shot.title,
            shot_description=shot.description, shot_intent=shot.intent,
            questions=tuple(SemanticQuestion(constraint_id=c.id, text=c.text, ask=c.ask) for c in slow),
            subjects={label: texts.name_of(label, names) for label in sorted(mentioned)},
            clip_dir=take.dir, fps=take.fps,
        )

    def _position_fact(self, establish: Establish, take: Take, names: Mapping[str, str],
                       now: float) -> Optional[SceneFact]:
        """Aggregate where the subject sits relative to the anchor from the take's per-frame detections. Only frames showing both count."""
        samples = []
        for frame in take.frames:
            world = World(frame.observation, {}, tuning=self._tuning)
            subject, anchor = world.live(establish.subject), world.live(establish.anchor)
            if subject is None or anchor is None:
                continue
            rel = relative_position(subject.box, anchor.box, self._tuning)
            if rel is not None:
                samples.append(rel)
        rel = aggregate(samples, self._tuning)
        if rel is None:
            return None

        # Confidence = share of frames showing both x whether left/right is clear. Below Tuning.confirm_below the UI flags "please confirm"
        coverage = len(samples) / len(take.frames)
        clarity = 1.0 if rel.side != "center" else 0.5
        return SceneFact(
            subject=establish.subject, field="position", value=rel.as_value(establish.anchor),
            label=texts.position_label(texts.name_of(establish.anchor, names), rel.vertical, rel.side),
            source=Source.YOLO, confidence=round(coverage * clarity, 2),
            take_id=take.take_id, ts=now,
        )
