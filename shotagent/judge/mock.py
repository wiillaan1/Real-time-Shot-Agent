"""Fake semantic judge: does not look at the picture, just returns pass / fail.

Everything it writes is prefixed with [mock] and the model field honestly says "mock",
so a placeholder description in the ledger is never mistaken for a real viewing.

To walk through "fail -> reshoot in place":
    tick "Make the mock fail the next take" in the UI (calls force_next), or pass a script at construction,
    e.g. MockJudge(script={"S02": [False, True]}) fails S02's first take and passes its second.

Dependencies: contracts, judge.base.
"""
from __future__ import annotations

import threading
import time
from typing import Mapping, Optional, Sequence

from ..contracts import JudgeAnswer, JudgeRequest, JudgeVerdict


class MockJudge:
    name = "mock"

    def __init__(self, latency_s: float = 0.0, script: Optional[Mapping[str, Sequence[bool]]] = None):
        self._latency_s = latency_s
        self._script = {shot_id: list(results) for shot_id, results in (script or {}).items()}
        self._forced: Optional[bool] = None
        self._lock = threading.Lock()
        self.requests: list[JudgeRequest] = []   # for tests: see what the slow loop actually asked

    def force_next(self, passed: Optional[bool]) -> None:
        """Set the result of the next judgement (once only). None = cancel."""
        with self._lock:
            self._forced = passed

    @property
    def forced(self) -> Optional[bool]:
        return self._forced

    def judge(self, request: JudgeRequest) -> JudgeVerdict:
        started = time.perf_counter()
        if self._latency_s > 0:
            time.sleep(self._latency_s)   # pretend to watch the take, so the UI can sit in "checking"
        with self._lock:
            self.requests.append(request)
            if self._forced is not None:
                passed, self._forced = self._forced, None
            elif self._script.get(request.shot_id):
                passed = self._script[request.shot_id].pop(0)
            else:
                passed = True

        reason = "[mock] did not actually watch the take; set to " + ("pass" if passed else "fail")
        return JudgeVerdict(
            answers=tuple(
                JudgeAnswer(constraint_id=q.constraint_id, passed=passed, reason=reason)
                for q in request.questions
            ),
            description=f"[mock] the semantic model's description of the take \"{request.shot_title}\" will go here",
            subject_notes={
                label: f"[mock] placeholder description of the {name}" for label, name in request.subjects.items()
            } if passed else {},
            model="mock",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
