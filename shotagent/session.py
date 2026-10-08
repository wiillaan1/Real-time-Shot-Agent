"""Session: the agent loop itself. It strings the two loops, the ledger and the user's actions into one state machine.

    FRAMING --roll--> RECORDING --end / timer runs out--> CHECKING
        ^                                                    |
        |          passed -> next shot (or DONE)             |
        +----------failed -> same shot, reshoot in place <---+

The design doc's suggested modules do not include it, but someone has to answer these questions, so it gets its own file:
  - Which shot is being filmed?          Not stored; derived from the ledger each time: the manually selected one, else the first not yet passed
  - Who gets this frame?                 Framing / recording -> the fast loop; while recording the frame also goes into the take
  - When is a take settled?              On manual end, or when the fixed timer runs out -> the slow loop (in the background, frames keep flowing)
  - When does the fast loop get a new snapshot?   Only in _enter_framing(): the moment a shot starts framing, never during a take

The product rule "pass before you leave the setup" also lives here: while checking, the prompt says to wait in place;
only after a pass does it say to move (or that the setup stays the same); after a fail it stays on the same shot.

The session never writes the ledger: all it holds are functions that read it. To write, it can only ask the planner
to reload the list or hand a take to the slow loop for settlement.

Changing the shot list (replan) is a side branch of the state machine: allowed while framing or when everything is done.
During it the session sits in PLANNING (waiting for the model to break the idea / script into shots) and returns to framing either way.

Concurrency: Session methods are called on the event-loop thread (which is why the HTTP routes are all async def);
only the heavy work goes to worker threads: fast.step, takes.save, slow.settle, planner.draft.

Dependencies: contracts, config, texts, fast_loop, slow_loop, takes, planner, ledger.state.
Never: ledger.store, ledger.events.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Optional

from . import texts
from .config import Settings
from .contracts import Frame, SourceCaps
from .fast_loop import FastLoop
from .ledger.state import LedgerState
from .planner import PlanCheck, PlanError, Planner
from .slow_loop import Settlement, SlowLoop
from .takes import TakeBuffer, TakeStore

log = logging.getLogger(__name__)


class Phase(str, Enum):
    PLANNING = "planning"      # writing the shot list: the planner is waiting for the model, no shooting meanwhile
    FRAMING = "framing"        # framing: the fast loop gives hints, waiting for the user to roll
    RECORDING = "recording"    # recording: the fast loop keeps hinting while the take is collected
    CHECKING = "checking"      # checking: the slow loop is settling, the user waits in place
    DONE = "done"              # every shot has passed


class SessionError(Exception):
    """The user cannot do this right now. status is the HTTP status code for the server layer."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


@dataclass
class Notice:
    """A message for the user, usually the result of the last take. It stays until the next result replaces it
    (or the user starts over / switches shots), so while reshooting you can still see why the last take failed."""

    kind: str          # pass | fail | error | info
    text: str
    take_id: str = ""
    at: float = 0.0


class Session:
    def __init__(
        self,
        *,
        read_ledger: Callable[[], LedgerState],        # read-only: Ledger.snapshot
        read_log: Callable[[int], list[dict[str, Any]]],   # read-only: Ledger.recent
        planner: Planner,
        fast: FastLoop,
        slow: SlowLoop,
        takes: TakeStore,
        settings: Settings,
        info: Callable[[], dict[str, Any]] = dict,     # other runtime info for the UI's status line
    ):
        self._read_ledger = read_ledger
        self._read_log = read_log
        self._planner = planner
        self._fast = fast
        self._slow = slow
        self._takes = takes
        self._settings = settings
        self._info = info

        self.phase = Phase.FRAMING
        self._shot_id: Optional[str] = None      # current shot
        self._selected: Optional[str] = None     # shot picked by hand (to reshoot one that already passed)
        self._caps: Optional[SourceCaps] = None
        self._buffer: Optional[TakeBuffer] = None
        self._duration_s = 0.0                   # fixed timer length of the take being recorded
        self._timer: Optional[asyncio.TimerHandle] = None
        self._settling: Optional[asyncio.Task] = None
        self._checking: Optional[dict[str, Any]] = None
        self._planning: Optional[dict[str, Any]] = None
        self._notice: Optional[Notice] = None
        self._busy = False                       # a frame is being processed
        self._rev = 0                            # the UI refetches the full state when this changes

    # ───────────── Boot / start over ─────────────

    def boot(self) -> None:
        """Called once at server start: have the planner load a list if the ledger has none, then enter framing."""
        outcome = self._planner.ensure()
        if outcome is PlanCheck.RELOADED:
            log.info("The shot list definition changed and the ledger holds no progress yet: switched to the new list")
        elif outcome is PlanCheck.STALE:
            log.warning(texts.PLAN_CHANGED)
            self._notice = Notice("info", texts.PLAN_CHANGED, at=time.time())
        self._enter_framing()

    def reset(self) -> None:
        if self.phase is Phase.CHECKING:
            raise SessionError("A take is being checked: wait for it to finish before starting over")
        if self.phase is Phase.PLANNING:
            raise SessionError("The shot list is being written: wait for it to finish before starting over")
        self._discard_take()
        self._selected = None
        self._notice = None
        self._planner.restart()  # the planner rewrites the same list: shot log and scene state are cleared
        self._enter_framing()

    async def replan(self, request: str) -> None:
        """Switch to another shot list: request is an idea or a script; empty = back to the hand-written example.

        Generation waits on the model (ten seconds or more), so it runs in a worker thread; meanwhile the session sits in PLANNING and no take can start.
        If it fails (no model configured, call failed, validation failed) the ledger is untouched and the session returns to the shot it was on.
        """
        if self.phase not in (Phase.FRAMING, Phase.DONE):
            raise SessionError("The shot list cannot be changed right now: " + texts.BANNER[self.phase.value])
        self._discard_take()
        roles = self._fast.describe()["roles"]
        self.phase = Phase.PLANNING
        self._planning = {"request": request.strip()[:120], "model": self._planner.generator_name}
        self._touch()
        try:
            plan = await asyncio.to_thread(
                self._planner.draft, request, frozenset(roles) if roles is not None else None)
        except PlanError as exc:
            log.warning("The shot list was not generated: %s", exc)
            raise SessionError(f"The shot list was not generated: {exc}", status=502) from exc
        else:
            self._planner.commit(plan)
            self._selected = None
            self._notice = Notice("info", f"Switched to the new list \"{plan.style}\": {len(plan.shots)} shots", at=time.time())
        finally:   # back to framing either way: the new list's first shot on success, the previous shot on failure
            self._planning = None
            self._enter_framing()

    # ───────────── Video source ─────────────

    def set_source(self, caps: SourceCaps) -> None:
        self._caps = caps
        if self.phase is Phase.RECORDING:
            self._discard_take()
            self._notice = Notice("error", "The video source changed, so that take is void. Please reshoot", at=time.time())
        if self.phase not in (Phase.CHECKING, Phase.PLANNING):
            self._enter_framing()    # capabilities changed: which constraints can be checked automatically must be recomputed
        self._touch()

    # ───────────── Every frame ─────────────

    async def on_frame(self, frame: Frame) -> dict[str, Any]:
        if self._busy:
            # The previous frame is still being processed: drop this one. Live hints only care about the latest picture, queueing is pointless
            return {"seq": frame.seq, "dropped": True, "guidance": None, **self._status()}
        self._busy = True
        try:
            guidance = None
            if self.phase in (Phase.FRAMING, Phase.RECORDING):
                buffer = self._buffer
                guidance = await asyncio.to_thread(self._fast.step, frame)
                # Only frames that arrived during the take and finished processing during the same take join it
                if buffer is not None and self._buffer is buffer and self.phase is Phase.RECORDING:
                    buffer.add(frame, guidance)
            return {
                "seq": frame.seq,
                "dropped": False,
                "guidance": guidance.model_dump(mode="json") if guidance else None,
                **self._status(),
            }
        finally:
            self._busy = False

    # ───────────── User actions ─────────────

    def start_take(self) -> None:
        if self.phase is not Phase.FRAMING or self._shot_id is None:
            raise SessionError("Cannot roll right now: " + texts.BANNER[self.phase.value])
        if self._caps is None:
            raise SessionError("There is no video source yet")
        record = self._read_ledger().record(self._shot_id)
        if record is None:
            raise SessionError(f"The ledger has no shot {self._shot_id}", status=404)

        take_id = f"{record.shot.id}-T{record.attempts + 1}"
        self._buffer = TakeBuffer(take_id, record.shot)
        self._duration_s = self._settings.take_seconds or record.shot.duration_s
        self.phase = Phase.RECORDING
        # Fixed timer: the take ends by itself when it runs out (the user may end it earlier)
        self._timer = asyncio.get_running_loop().call_later(
            self._duration_s, self._on_timer, self._buffer)
        self._touch()

    def stop_take(self, reason: str = "manual") -> None:
        if self.phase is not Phase.RECORDING or self._buffer is None:
            raise SessionError("No take is being recorded")
        buffer, self._buffer = self._buffer, None
        self._cancel_timer()
        ended_at = time.time()

        if buffer.n_frames < self._settings.min_take_frames:
            self._notice = Notice(
                "error", f"This take has only {buffer.n_frames} frames: too short, so it was not checked. Please reshoot",
                take_id=buffer.take_id, at=ended_at)
            self._enter_framing()
            return

        self.phase = Phase.CHECKING
        self._checking = {"take_id": buffer.take_id, "shot_id": buffer.shot.id, "frames": buffer.n_frames}
        self._touch()
        self._settling = asyncio.get_running_loop().create_task(self._settle(buffer, ended_at, reason))

    def select_shot(self, shot_id: Optional[str]) -> None:
        """Pick which shot to film by hand (e.g. reshoot one that passed). None = back to the automatic order."""
        if self.phase not in (Phase.FRAMING, Phase.DONE):
            raise SessionError("Cannot switch shots while recording or checking")
        if shot_id is not None and self._read_ledger().record(shot_id) is None:
            raise SessionError(f"There is no shot {shot_id}", status=404)
        self._selected = shot_id
        self._notice = None
        self._enter_framing()

    # ───────────── State machine internals ─────────────

    def _enter_framing(self) -> None:
        """Start framing a shot. This is where, and the only place where, the fast loop gets a fresh ledger snapshot."""
        state = self._read_ledger()
        if self._selected is not None and state.record(self._selected) is None:
            self._selected = None
        self._shot_id = self._selected or state.next_shot_id()
        self.phase = Phase.FRAMING if self._shot_id else Phase.DONE
        self._fast.load(state, self._shot_id, self._caps)
        self._touch()

    def _on_timer(self, buffer: TakeBuffer) -> None:
        if self._buffer is buffer and self.phase is Phase.RECORDING:
            self.stop_take("timer")

    async def _settle(self, buffer: TakeBuffer, ended_at: float, reason: str) -> None:
        """Background: save the take -> slow-loop settlement -> back to framing according to the result."""
        try:
            take = await asyncio.to_thread(self._takes.save, buffer, ended_at, reason)
            result = await asyncio.to_thread(self._slow.settle, take)
        except Exception as exc:   # e.g. the semantic model is unreachable. Settlement is atomic: no commit means the ledger did not change
            log.exception("Settling take %s failed", buffer.take_id)
            self._notice = Notice(
                "error", f"The check could not be done: {exc}. The ledger is unchanged, please reshoot", take_id=buffer.take_id, at=time.time())
        else:
            self._notice = self._notice_for(result)
        finally:   # always return to framing so the state machine never gets stuck in "checking"
            self._selected = None
            self._checking = None
            self._enter_framing()

    def _notice_for(self, result: Settlement) -> Notice:
        state = result.state
        shot = state.record(result.record.shot_id).shot
        now = time.time()

        if not result.passed:
            reasons = "; ".join(result.reasons) or "it did not pass the check"
            if result.kept_previous:
                text = f"{shot.id}: this take did not pass ({reasons}). The earlier passing take is kept"
            else:
                text = f"{shot.id} did not pass: {reasons}. Stay where you are and reshoot"
            return Notice("fail", text, take_id=result.record.take_id, at=now)

        next_id = state.next_shot_id()
        if next_id is None:
            text = f"{shot.id} passed. All shots are done"
        else:
            following = state.record(next_id).shot
            if following.setup and following.setup == shot.setup:
                move = f"Same setup: next is {following.id} \"{following.title}\""
            else:
                move = f"You can move to {following.setup} now: next is {following.id} \"{following.title}\""
            text = f"{shot.id} passed. {move}"
        return Notice("pass", text, take_id=result.record.take_id, at=now)

    def _discard_take(self) -> None:
        self._buffer = None
        self._cancel_timer()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _touch(self) -> None:
        self._rev += 1

    # ───────────── For the UI ─────────────

    def _take_progress(self) -> Optional[dict[str, Any]]:
        buffer = self._buffer
        if buffer is None:
            return None
        return {
            "take_id": buffer.take_id,
            "shot_id": buffer.shot.id,
            "elapsed_s": round(buffer.elapsed(), 2),
            "duration_s": self._duration_s,
            "frames": buffer.n_frames,
        }

    def _status(self) -> dict[str, Any]:
        """Brief status sent back with every frame reply. When rev changes, the UI fetches the full view()."""
        return {"rev": self._rev, "phase": self.phase.value, "take": self._take_progress()}

    def view(self) -> dict[str, Any]:
        """Full state: session + ledger. The UI's shot list, scene state and write log all come from here."""
        state = self._read_ledger()
        passed, total = state.coverage()
        fast = self._fast.describe()
        return {
            **self._status(),
            "banner": texts.BANNER[self.phase.value],
            "current_shot_id": self._shot_id,
            "manual_selection": self._selected is not None,
            "notice": asdict(self._notice) if self._notice else None,
            "checking": self._checking,
            "planning": self._planning,
            "briefing": fast["briefing"],
            "modes": fast["modes"],
            "runtime": {
                "detector": fast["detector"],
                "detector_note": fast["detector_note"],
                "capabilities": fast["capabilities"],
                "source": self._caps.model_dump() if self._caps else None,
                "confirm_below": self._settings.tuning.confirm_below,
                **self._info(),
            },
            "coverage": {"passed": passed, "total": total},
            # Scene facts with low confidence: the UI flags "please confirm" (minimal conflict handling; the confirm action itself is not built yet)
            "confirm": [f"{f.subject}.{f.field}" for f in state.scene
                        if f.confidence < self._settings.tuning.confirm_below],
            "ledger": state.model_dump(mode="json"),
            "log": self._read_log(12),
        }
