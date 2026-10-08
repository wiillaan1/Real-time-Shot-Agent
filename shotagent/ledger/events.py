"""Write events: the only three things that can change the ledger.

    PlanLoaded          planner: load (or reset to) a shot list
    TakeSettled         slow loop: a take has been settled
    SceneFactObserved   slow loop: a scene fact derived from a kept take

Who may send which event: store.WRITE_PERMISSIONS. How an event changes the state: reducer.apply.

Dependencies: contracts, ledger.state.
"""
from __future__ import annotations

from typing import Literal, Union

from ..contracts import Frozen, Plan
from .state import SceneFact, TakeRecord


class PlanLoaded(Frozen):
    kind: Literal["plan_loaded"] = "plan_loaded"
    plan: Plan


class TakeSettled(Frozen):
    kind: Literal["take_settled"] = "take_settled"
    take: TakeRecord


class SceneFactObserved(Frozen):
    kind: Literal["scene_fact_observed"] = "scene_fact_observed"
    fact: SceneFact


Event = Union[PlanLoaded, TakeSettled, SceneFactObserved]
