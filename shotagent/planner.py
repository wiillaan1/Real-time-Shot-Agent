"""Planner: turn the user's request into a shot list and write it to the ledger.

    make_plan(request)   request -> Plan
        empty request     the hand-written example list (plans.py: Hitchcock's "bag under the table")
        idea or script    handed to the generator (plan_gen.PlanGenerator, an LLM behind it), which breaks it into shots with checkable constraints

Either way the output is a Plan; downstream (ledger, both loops, UI) neither knows nor cares whether it was hand-written or generated.

Planner splits generating from writing, because generating waits on a model for seconds and may fail:
    draft(request)   generate only, do not touch the ledger (the session runs it in a worker thread)
    commit(plan)     write to the ledger: the list is replaced, shot log and scene state are cleared
    load(request)    both steps in a row
    restart()        clear progress, keep the list ("Start over"): a generated list is not sent to the model again
    ensure()         at server start: make sure the ledger has a list, without touching existing progress

Ledger writes only go through Ledger.commit(..., writer="planner"), and the only event allowed is PlanLoaded.

Dependencies: contracts, plans, plan_gen, ledger.store, ledger.events.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from .contracts import Plan
from .ledger.events import PlanLoaded
from .ledger.store import Ledger
from .plan_gen import PlanError
from .plans import hitchcock_bag_under_table

__all__ = ["Generate", "PlanCheck", "PlanError", "Planner", "make_plan"]

# Generator: (idea or script, roles allowed this time; None = all) -> Plan. Raises PlanError when it cannot
Generate = Callable[[str, Optional[frozenset[str]]], Plan]

NO_GENERATOR = ("No model is configured for writing shot lists: set the OPENAI_API_KEY environment variable and restart run.py"
                " (only the example list is available for now)")


class PlanCheck(str, Enum):
    """The outcome of Planner.ensure()."""

    LOADED = "loaded"        # the ledger was empty, the example list was loaded
    KEPT = "kept"            # the list in the ledger stays, carry on shooting
    RELOADED = "reloaded"    # the example list's definition changed and the ledger holds no take yet -> replaced with the new one
    STALE = "stale"          # the example list's definition changed but there is progress -> ledger untouched, the user decides whether to start over


def make_plan(request: str = "", generate: Optional[Generate] = None,
              roles: Optional[frozenset[str]] = None) -> Plan:
    """Request -> shot list. An empty request gives the hand-written example; anything else goes to the generator."""
    request = (request or "").strip()
    if not request:
        return hitchcock_bag_under_table()
    if generate is None:
        raise PlanError(NO_GENERATOR)
    return generate(request, roles)


class Planner:
    def __init__(self, ledger: Ledger, generate: Optional[Generate] = None):
        self._ledger = ledger
        self._generate = generate

    @property
    def generator_name(self) -> Optional[str]:
        """Name of the model that writes shot lists; None when not configured (the UI uses it to enable the idea box)."""
        if self._generate is None:
            return None
        return getattr(self._generate, "name", "") or "LLM"

    def draft(self, request: str = "", roles: Optional[frozenset[str]] = None) -> Plan:
        """Generate only, no ledger write. May wait on the model for ten seconds or more, and may raise PlanError."""
        return make_plan(request, self._generate, roles)

    def commit(self, plan: Plan) -> None:
        """Write the list to the ledger. Clears the existing shot log and scene state (i.e. starts afresh)."""
        self._ledger.commit([PlanLoaded(plan=plan)], writer="planner")

    def load(self, request: str = "", roles: Optional[frozenset[str]] = None) -> Plan:
        plan = self.draft(request, roles)
        self.commit(plan)
        return plan

    def restart(self) -> Plan:
        """Clear progress and start again from the first shot. The list stays:
        a generated list is kept as is (the model is not asked again, it would not write the same one); the example list takes the latest definition in code."""
        state = self._ledger.snapshot()
        if state.shots and state.request:
            plan = Plan(
                style=state.style, subjects=dict(state.subjects), shots=tuple(r.shot for r in state.shots),
                request=state.request, synopsis=state.synopsis,
            )
        else:
            plan = make_plan()
        self.commit(plan)
        return plan

    def ensure(self) -> PlanCheck:
        """Called once at server start: make sure the ledger has a list without clearing progress (carry on after a restart).

        It also handles an easy trap: edit plans.py, restart, and the ledger still holds the old example list.
        With nothing shot yet it is simply replaced; with progress it is left alone and STALE lets the UI tell the user.
        Generated lists skip this comparison: they are not in the code at all, the copy in the ledger is the only one.
        """
        state = self._ledger.snapshot()
        if not state.shots:
            self.load()
            return PlanCheck.LOADED
        if state.request:
            return PlanCheck.KEPT

        plan = make_plan()
        current = tuple(sorted(plan.shots, key=lambda shot: shot.order))
        stored = tuple(record.shot for record in state.shots)
        if stored == current and state.style == plan.style and state.subjects == plan.subjects:
            return PlanCheck.KEPT
        if all(record.attempts == 0 for record in state.shots):
            self.load()
            return PlanCheck.RELOADED
        return PlanCheck.STALE
