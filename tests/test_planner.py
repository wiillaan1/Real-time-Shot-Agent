"""Planner: loading the list, and what happens when plans.py is edited and the server restarted."""
from __future__ import annotations

from conftest import BAG_LEFT

from shotagent import planner as planner_module
from shotagent import texts
from shotagent.contracts import Plan, ShotStatus
from shotagent.ledger.events import TakeSettled
from shotagent.ledger.state import TakeRecord
from shotagent.ledger.store import Ledger
from shotagent.planner import PlanCheck, Planner
from shotagent.plans import hitchcock_bag_under_table


def edited_plan() -> Plan:
    """As if someone edited plans.py: S02's distance threshold goes from 0.5 to 0.8."""
    plan = hitchcock_bag_under_table()
    s02 = plan.shots[1]
    apart = s02.constraints[1]
    assert apart.predicate == "apart"
    tighter = apart.model_copy(update={"args": {**apart.args, "min_ratio": 0.8}})
    new_s02 = s02.model_copy(update={"constraints": (s02.constraints[0], tighter) + s02.constraints[2:]})
    return plan.model_copy(update={"shots": (plan.shots[0], new_s02, plan.shots[2])})


def min_ratio(ledger: Ledger) -> float:
    return ledger.snapshot().record("S02").shot.constraints[1].args["min_ratio"]


def settle_s01(ledger: Ledger) -> None:
    ledger.commit([TakeSettled(take=TakeRecord(
        take_id="S01-T1", shot_id="S01", started_at=0, ended_at=5, n_frames=10, clip_dir="/clips", passed=True,
    ))], writer="slow_loop")


def test_loads_the_plan_into_an_empty_ledger(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    assert Planner(ledger).ensure() is PlanCheck.LOADED
    state = ledger.snapshot()
    assert [r.shot.id for r in state.shots] == ["S01", "S02", "S03"] and state.subjects["bag"] == "bag"


def test_restart_keeps_progress_when_the_plan_is_unchanged(tmp_path):
    first = Ledger(tmp_path / "ledger.json")
    Planner(first).ensure()
    settle_s01(first)

    restarted = Ledger(tmp_path / "ledger.json")          # restart: the list read back from JSON equals the definition in code field by field
    assert Planner(restarted).ensure() is PlanCheck.KEPT
    assert restarted.snapshot().record("S01").status is ShotStatus.PASSED


def test_edited_plan_replaces_the_stored_one_when_nothing_has_been_shot(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "ledger.json")
    Planner(ledger).ensure()
    assert min_ratio(ledger) == 0.5

    monkeypatch.setattr(planner_module, "make_plan", lambda *args, **kwargs: edited_plan())
    assert Planner(Ledger(tmp_path / "ledger.json")).ensure() is PlanCheck.RELOADED
    assert min_ratio(Ledger(tmp_path / "ledger.json")) == 0.8


def test_edited_plan_is_not_forced_onto_a_ledger_with_progress(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "ledger.json")
    Planner(ledger).ensure()
    settle_s01(ledger)

    monkeypatch.setattr(planner_module, "make_plan", lambda *args, **kwargs: edited_plan())
    restarted = Ledger(tmp_path / "ledger.json")
    assert Planner(restarted).ensure() is PlanCheck.STALE
    assert min_ratio(restarted) == 0.5                                    # ledger untouched: progress is there, the list is the old one
    assert restarted.snapshot().record("S01").status is ShotStatus.PASSED

    Planner(restarted).load()                                             # it only changes when the user starts over
    assert min_ratio(restarted) == 0.8 and restarted.snapshot().coverage() == (0, 3)


def test_the_interface_says_so_when_the_stored_plan_is_stale(make_rig, tmp_path, monkeypatch):
    first = make_rig(data_dir=tmp_path / "shared")
    first.shoot(BAG_LEFT)
    assert first.state()["notice"]["kind"] == "pass"

    monkeypatch.setattr(planner_module, "make_plan", lambda *args, **kwargs: edited_plan())
    second = make_rig(data_dir=tmp_path / "shared")
    state = second.state()
    assert (state["notice"]["kind"], state["notice"]["text"]) == ("info", texts.PLAN_CHANGED)
    assert state["current_shot_id"] == "S02" and state["coverage"]["passed"] == 1      # progress as it was

    fresh = second.post("/api/reset").json()                              # start over: the new list comes in and the notice goes away
    assert fresh["notice"] is None and fresh["coverage"]["passed"] == 0
    assert fresh["ledger"]["shots"][1]["shot"]["constraints"][1]["args"]["min_ratio"] == 0.8
