"""The ledger's rules: state + event -> new state (pure functions).

All of the ledger's invariants are in this one file, and no writer can get around them:

  1. Each shot keeps exactly one take (design doc 2.5, item 2):
       new take passes                       -> the new one is kept, status passed
       new take fails, old one passed        -> the old one stays, status still passed
       both fail (or there is no old one)    -> the latest is kept, status retake
  2. When a take is replaced, the scene facts derived from it are dropped (the new facts arrive in the same commit),
     so every scene fact in the ledger traces back to a take that is currently kept and passed.
  3. Conflicts are settled by a fixed source per field, never by weighted averaging (design doc 2.4):
       position trusts YOLO, description trusts Cosmos.
     A non-authoritative source may fill an empty field but may not overwrite what the authority wrote.

Dependencies: contracts, ledger.state, ledger.events.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts import ShotStatus, Source
from .events import Event, PlanLoaded, SceneFactObserved, TakeSettled
from .state import LedgerError, LedgerState, ShotRecord

# Field -> the source that has the final say
FIELD_AUTHORITY: dict[str, Source] = {
    "position": Source.YOLO,
    "description": Source.COSMOS,
}


_FIELD_NAME = {"position": "position", "description": "description"}   # field names as shown in the write log


@dataclass(frozen=True)
class Applied:
    state: LedgerState
    changed: bool      # False = a valid event the rules ignored (still logged, for traceability)
    summary: str       # what was written
    note: str          # how the rules handled it


def apply(state: LedgerState, event: Event) -> Applied:
    if isinstance(event, PlanLoaded):
        return _plan_loaded(state, event)
    if isinstance(event, TakeSettled):
        return _take_settled(state, event)
    if isinstance(event, SceneFactObserved):
        return _scene_fact_observed(state, event)
    raise LedgerError(f"Unknown event: {type(event).__name__}")


def _plan_loaded(state: LedgerState, event: PlanLoaded) -> Applied:
    plan = event.plan
    ids = [s.id for s in plan.shots]
    if len(set(ids)) != len(ids):
        raise LedgerError("The shot list has duplicate shot ids")
    shots = tuple(ShotRecord(shot=s) for s in sorted(plan.shots, key=lambda s: s.order))
    new = state.model_copy(update={
        "style": plan.style,
        "request": plan.request,
        "synopsis": plan.synopsis,
        "subjects": dict(plan.subjects),
        "shots": shots,
        "scene": (),
    })
    return Applied(new, True, f"Loaded shot list \"{plan.style}\"", f"{len(shots)} shots; shot log and scene state cleared")


def _take_settled(state: LedgerState, event: TakeSettled) -> Applied:
    take = event.take
    rec = state.record(take.shot_id)
    if rec is None:
        raise LedgerError(f"The ledger has no shot {take.shot_id}")

    old = rec.effective_take
    if take.passed:
        effective, status = take, ShotStatus.PASSED
        note = "passed, now the kept take" + (f" (replaces {old.take_id})" if old else "")
    elif old is not None and old.passed:
        effective, status = old, ShotStatus.PASSED
        note = f"failed, the kept take {old.take_id} stays"
    else:
        effective, status = take, ShotStatus.RETAKE
        note = "failed, kept as the latest take, status: retake"

    new_rec = rec.model_copy(update={
        "status": status,
        "effective_take": effective,
        "attempts": rec.attempts + 1,
        "updated_at": take.settled_at,
    })
    scene = state.scene
    if old is not None and effective is not old:
        scene = tuple(f for f in scene if f.take_id != old.take_id)

    new = state.model_copy(update={
        "shots": tuple(new_rec if r.shot.id == take.shot_id else r for r in state.shots),
        "scene": scene,
    })
    return Applied(new, True, f"{take.take_id} settled: {'passed' if take.passed else 'failed'}", note)


def _scene_fact_observed(state: LedgerState, event: SceneFactObserved) -> Applied:
    fact = event.fact
    subject = state.subjects.get(fact.subject, fact.subject)
    summary = f"{subject} {_FIELD_NAME.get(fact.field, fact.field)}: {fact.label}"

    take = state.find_take(fact.take_id)
    if take is None or not take.passed:
        raise LedgerError(f"A scene fact must come from a kept take that passed; {fact.take_id} is not one")

    authority = FIELD_AUTHORITY.get(fact.field)
    existing = state.fact(fact.subject, fact.field)
    if (
        existing is not None
        and authority is not None
        and existing.source is authority
        and fact.source is not authority
    ):
        return Applied(
            state, False, summary,
            f"ignored: the {fact.field} field follows {authority.value}, {fact.source.value} may not overwrite it",
        )

    scene = tuple(
        f for f in state.scene if not (f.subject == fact.subject and f.field == fact.field)
    ) + (fact,)
    note = "updated" if existing is not None else "added"
    return Applied(state.model_copy(update={"scene": scene}), True, summary, note)
