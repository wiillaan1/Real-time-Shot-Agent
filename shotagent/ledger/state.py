"""The ledger's state (the read model).

Two layers (design doc 2.4):
    shots  shot log: each shot's plan, status, kept take, and the verdict on every constraint
    scene  scene state: objects with their relative positions / semantic descriptions

Every conclusion carries a source tag:
    ShotRecord.shot                from the plan (Shot.source = plan)
    ConstraintVerdict.source       yolo / sensor / cosmos, per constraint
    TakeRecord.description         from Cosmos (the judge field records which model it really was; the mock writes "mock")
    SceneFact.source               position facts from yolo, semantic ones from cosmos

Everything here is immutable. The "ledger snapshot" the fast loop receives is a LedgerState.

Dependencies: contracts only.
"""
from __future__ import annotations

from typing import Any, Optional

from ..contracts import Frozen, Shot, ShotStatus, Source


class LedgerError(Exception):
    """A ledger write was refused (no permission, invalid event, broken invariant)."""


class ConstraintVerdict(Frozen):
    """The verdict on one constraint for one take."""

    constraint_id: str
    text: str
    passed: Optional[bool]           # None = not checked automatically (degraded for lack of a capability, or a fast constraint failed so the semantic model was not asked)
    source: Source
    detail: str = ""
    score: Optional[float] = None    # fast constraints: share of satisfying frames; semantic ones: the model's confidence


class TakeRecord(Frozen):
    """The settlement of one take. Only the kept take stays in the ledger; replaced files stay on disk but leave the index."""

    take_id: str
    shot_id: str
    started_at: float
    ended_at: float
    n_frames: int
    clip_dir: str
    passed: bool
    verdicts: tuple[ConstraintVerdict, ...] = ()
    description: str = ""            # semantic description (source cosmos)
    judge: str = ""                  # the model that actually gave the semantic judgement
    settled_at: float = 0.0


class ShotRecord(Frozen):
    shot: Shot                       # the plan (source plan)
    status: ShotStatus = ShotStatus.TODO
    attempts: int = 0                # how many takes have been settled
    effective_take: Optional[TakeRecord] = None   # each shot keeps exactly one take
    updated_at: float = 0.0


class SceneFact(Frozen):
    """One fact in the scene state, e.g. bag.position = under the table, left side (yolo, from S01-T1)."""

    subject: str                     # role name, e.g. "bag"
    field: str                       # "position" (relative position) | "description" (semantic description)
    value: dict[str, Any]            # position: spatial.RelPos.as_value(); description: {"text": ...}
    label: str                       # a phrase for people, e.g. "under the table, left side"
    source: Source
    confidence: float = 1.0
    take_id: str                     # the kept take this fact was derived from
    ts: float = 0.0


class LedgerState(Frozen):
    version: int = 0                 # incremented on every commit
    style: str = ""
    request: str = ""                # the idea / script this list came from; empty = the hand-written example
    synopsis: str = ""               # story synopsis
    subjects: dict[str, str] = {}    # role name -> display name
    shots: tuple[ShotRecord, ...] = ()
    scene: tuple[SceneFact, ...] = ()
    updated_at: float = 0.0

    # -- Read-only queries --

    def record(self, shot_id: str) -> Optional[ShotRecord]:
        return next((r for r in self.shots if r.shot.id == shot_id), None)

    def fact(self, subject: str, field: str) -> Optional[SceneFact]:
        return next((f for f in self.scene if f.subject == subject and f.field == field), None)

    def next_shot_id(self) -> Optional[str]:
        """The first shot in plan order that has not passed yet. None when all have passed."""
        for rec in sorted(self.shots, key=lambda r: r.shot.order):
            if rec.status is not ShotStatus.PASSED:
                return rec.shot.id
        return None

    def find_take(self, take_id: str) -> Optional[TakeRecord]:
        """Find a take in the index (i.e. among the shots' kept takes)."""
        for rec in self.shots:
            if rec.effective_take is not None and rec.effective_take.take_id == take_id:
                return rec.effective_take
        return None

    def coverage(self) -> tuple[int, int]:
        """(number of shots passed, total number of shots)."""
        return sum(1 for r in self.shots if r.status is ShotStatus.PASSED), len(self.shots)
