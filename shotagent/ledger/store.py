"""Ledger: the ledger's state manager and its single write path.

    read:  snapshot()  returns an immutable snapshot (a deep copy); anyone may call it
    write: commit()    the only function in the project that can change the ledger

Who may write what is fixed in WRITE_PERMISSIONS: the planner may only load a plan, the slow loop may only settle takes
and write scene facts. The fast loop is not on the list, and it never even gets a Ledger object (see wiring.py), only snapshots.

A commit is atomic: if any event in it is invalid, none of it takes effect.
Every commit bumps the version and appends to *.events.jsonl, so each field in the ledger can be traced to
who wrote it, when, and because of which take.

Dependencies: contracts, ledger.state, ledger.events, ledger.reducer.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import reducer
from .events import Event, PlanLoaded, SceneFactObserved, TakeSettled
from .state import LedgerError, LedgerState

log = logging.getLogger(__name__)

# Write permissions: writer -> the event types it may commit
WRITE_PERMISSIONS: dict[str, tuple[type, ...]] = {
    "planner": (PlanLoaded,),
    "slow_loop": (TakeSettled, SceneFactObserved),
}


class Ledger:
    def __init__(self, path: Optional[Path] = None, clock: Callable[[], float] = time.time):
        """With path None the ledger lives in memory only (for tests)."""
        self._lock = threading.RLock()
        self._clock = clock
        self._path = Path(path) if path else None
        self._log_path = self._path.with_name(self._path.stem + ".events.jsonl") if self._path else None
        self._recent: deque[dict[str, Any]] = deque(maxlen=50)
        self._state = self._load()

    # ───────────── Read ─────────────

    def snapshot(self) -> LedgerState:
        """An immutable snapshot of the ledger as it is now. Later changes to the ledger do not affect it."""
        with self._lock:
            return self._state.model_copy(deep=True)

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        """The most recent write-log entries, newest first."""
        with self._lock:
            return list(self._recent)[-n:][::-1]

    # ───────────── Write (the only way in) ─────────────

    def commit(self, events: Sequence[Event], *, writer: str) -> LedgerState:
        allowed = WRITE_PERMISSIONS.get(writer)
        if allowed is None:
            raise LedgerError(f"{writer!r} has no permission to write the ledger")
        for event in events:
            if not isinstance(event, allowed):
                raise LedgerError(f"{writer!r} may not commit {type(event).__name__}")
        if not events:
            return self.snapshot()

        with self._lock:
            state = self._state
            results: list[tuple[Event, reducer.Applied]] = []
            for event in events:
                applied = reducer.apply(state, event)   # an invalid event raises LedgerError and voids the whole commit
                state = applied.state
                results.append((event, applied))

            now = self._clock()
            state = state.model_copy(update={"version": self._state.version + 1, "updated_at": now})
            entries = [self._entry(state.version, now, writer, ev, ap) for ev, ap in results]
            self._persist(state, entries, [ev for ev, _ in results])
            self._state = state
            self._recent.extend(entries)
            return state.model_copy(deep=True)

    # ───────────── Internals ─────────────

    @staticmethod
    def _entry(version: int, ts: float, writer: str, event: Event, applied: reducer.Applied) -> dict[str, Any]:
        return {
            "version": version,
            "ts": ts,
            "writer": writer,
            "kind": event.kind,
            "sources": _sources_of(event),
            "summary": applied.summary,
            "note": applied.note,
            "applied": applied.changed,
        }

    def _persist(self, state: LedgerState, entries: list[dict[str, Any]], events: list[Event]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self._path)   # atomic replace: a crash mid-write never leaves half a ledger
        with self._log_path.open("a", encoding="utf-8") as fh:
            for entry, event in zip(entries, events):
                fh.write(json.dumps({**entry, "event": event.model_dump(mode="json")}, ensure_ascii=False) + "\n")

    def _load(self) -> LedgerState:
        if self._path is None or not self._path.exists():
            return LedgerState()
        try:
            state = LedgerState.model_validate_json(self._path.read_text(encoding="utf-8"))
        except Exception as exc:   # schema changed or file corrupt: move the old file aside and start from an empty ledger
            aside = self._path.with_name(f"{self._path.name}.bad-{int(self._clock())}")
            os.replace(self._path, aside)
            log.warning("The ledger file could not be read (%s); moved it to %s and started from an empty ledger", exc, aside.name)
            return LedgerState()
        self._recent.extend(self._load_log_tail())
        return state

    def _load_log_tail(self) -> list[dict[str, Any]]:
        if self._log_path is None or not self._log_path.exists():
            return []
        entries = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines()[-50:]:
            try:
                entry = json.loads(line)
                entry.pop("event", None)
                entries.append(entry)
            except json.JSONDecodeError:
                continue
        return entries


def _sources_of(event: Event) -> list[str]:
    """Which sources this write used (for the write log)."""
    if isinstance(event, PlanLoaded):
        return sorted({shot.source.value for shot in event.plan.shots})
    if isinstance(event, TakeSettled):
        return sorted({v.source.value for v in event.take.verdicts})
    if isinstance(event, SceneFactObserved):
        return [event.fact.source.value]
    return []
