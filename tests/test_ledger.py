"""The ledger's rules: write permissions, the kept take, which source a field trusts, traceability, persistence."""
from __future__ import annotations

import json

import pytest

from shotagent.contracts import ShotStatus, Source
from shotagent.ledger.events import PlanLoaded, SceneFactObserved, TakeSettled
from shotagent.ledger.state import LedgerError, SceneFact, TakeRecord
from shotagent.ledger.store import Ledger
from shotagent.plans import hitchcock_bag_under_table


def take(take_id: str, passed: bool, shot_id: str = "S01") -> TakeSettled:
    return TakeSettled(take=TakeRecord(
        take_id=take_id, shot_id=shot_id, started_at=0, ended_at=5, n_frames=10,
        clip_dir=f"/clips/{take_id}", passed=passed, settled_at=6,
    ))


def fact(take_id: str, field: str = "position", source: Source = Source.YOLO,
         label: str = "under the table, left side", subject: str = "bag") -> SceneFactObserved:
    return SceneFactObserved(fact=SceneFact(
        subject=subject, field=field, value={"anchor": "table", "u": 0.2, "v": 0.85, "side": "left"},
        label=label, source=source, confidence=0.9, take_id=take_id,
    ))


@pytest.fixture
def ledger() -> Ledger:
    book = Ledger()
    book.commit([PlanLoaded(plan=hitchcock_bag_under_table())], writer="planner")
    return book


def test_plan_becomes_three_unshot_shots(ledger):
    state = ledger.snapshot()
    assert [r.shot.id for r in state.shots] == ["S01", "S02", "S03"]
    assert all(r.status is ShotStatus.TODO and r.effective_take is None for r in state.shots)
    assert all(r.shot.source is Source.PLAN for r in state.shots)
    assert state.next_shot_id() == "S01"
    assert state.coverage() == (0, 3)
    assert state.version == 1


# ───────────── Single write path: who may write what ─────────────

def test_only_listed_writers_may_commit(ledger):
    with pytest.raises(LedgerError):
        ledger.commit([take("S01-T1", True)], writer="fast_loop")
    with pytest.raises(LedgerError):
        ledger.commit([take("S01-T1", True)], writer="session")


def test_writers_are_limited_to_their_own_events(ledger):
    with pytest.raises(LedgerError):
        ledger.commit([take("S01-T1", True)], writer="planner")
    with pytest.raises(LedgerError):
        ledger.commit([PlanLoaded(plan=hitchcock_bag_under_table())], writer="slow_loop")
    assert ledger.snapshot().version == 1


def test_snapshot_is_isolated_from_later_writes(ledger):
    before = ledger.snapshot()
    ledger.commit([take("S01-T1", True)], writer="slow_loop")
    assert before.record("S01").status is ShotStatus.TODO
    assert ledger.snapshot().record("S01").status is ShotStatus.PASSED


def test_snapshot_cannot_be_used_to_change_the_ledger(ledger):
    snapshot = ledger.snapshot()
    with pytest.raises(Exception):
        snapshot.version = 99
    snapshot.subjects["bag"] = "changed"         # this changes the snapshot's own copy
    assert ledger.snapshot().subjects["bag"] == "bag"


# ───────────── One kept take per shot ─────────────

def test_passing_take_becomes_effective(ledger):
    ledger.commit([take("S01-T1", True)], writer="slow_loop")
    record = ledger.snapshot().record("S01")
    assert record.status is ShotStatus.PASSED
    assert record.effective_take.take_id == "S01-T1"
    assert record.attempts == 1


def test_new_pass_replaces_old_pass(ledger):
    ledger.commit([take("S01-T1", True)], writer="slow_loop")
    ledger.commit([take("S01-T2", True)], writer="slow_loop")
    assert ledger.snapshot().record("S01").effective_take.take_id == "S01-T2"


def test_failed_retake_keeps_the_old_passing_take(ledger):
    ledger.commit([take("S01-T1", True)], writer="slow_loop")
    ledger.commit([take("S01-T2", False)], writer="slow_loop")
    record = ledger.snapshot().record("S01")
    assert record.status is ShotStatus.PASSED
    assert record.effective_take.take_id == "S01-T1"
    assert record.attempts == 2


def test_two_failures_keep_the_latest_and_stay_retake(ledger):
    ledger.commit([take("S01-T1", False)], writer="slow_loop")
    ledger.commit([take("S01-T2", False)], writer="slow_loop")
    record = ledger.snapshot().record("S01")
    assert record.status is ShotStatus.RETAKE
    assert record.effective_take.take_id == "S01-T2"
    assert ledger.snapshot().next_shot_id() == "S01"


def test_take_for_unknown_shot_is_rejected(ledger):
    with pytest.raises(LedgerError):
        ledger.commit([take("S99-T1", True, shot_id="S99")], writer="slow_loop")


# ───────────── Scene state: source, priority, traceability ─────────────

def test_fact_carries_source_and_take(ledger):
    ledger.commit([take("S01-T1", True), fact("S01-T1")], writer="slow_loop")
    stored = ledger.snapshot().fact("bag", "position")
    assert (stored.source, stored.take_id, stored.label) == (Source.YOLO, "S01-T1", "under the table, left side")


def test_fact_must_trace_to_a_passing_effective_take(ledger):
    with pytest.raises(LedgerError):
        ledger.commit([fact("S01-T1")], writer="slow_loop")              # this take is not in the ledger at all
    with pytest.raises(LedgerError):
        ledger.commit([take("S01-T1", False), fact("S01-T1")], writer="slow_loop")   # a take that failed


def test_commit_is_atomic(ledger):
    """If one event in a commit is invalid, the valid one before it does not take effect either."""
    with pytest.raises(LedgerError):
        ledger.commit([take("S01-T1", True), fact("some other take")], writer="slow_loop")
    state = ledger.snapshot()
    assert state.version == 1
    assert state.record("S01").status is ShotStatus.TODO


def test_position_trusts_yolo_over_cosmos(ledger):
    ledger.commit([take("S01-T1", True), fact("S01-T1", source=Source.YOLO, label="under the table, left side")],
                  writer="slow_loop")
    ledger.commit([fact("S01-T1", source=Source.COSMOS, label="under the table, right side")], writer="slow_loop")
    stored = ledger.snapshot().fact("bag", "position")
    assert (stored.source, stored.label) == (Source.YOLO, "under the table, left side")
    ignored = ledger.recent(1)[0]
    assert ignored["applied"] is False and "yolo" in ignored["note"]     # the ignored write is logged as well


def test_non_authority_may_fill_an_empty_field_and_authority_overrides_it(ledger):
    ledger.commit([take("S01-T1", True), fact("S01-T1", source=Source.COSMOS, label="probably on the left")],
                  writer="slow_loop")
    assert ledger.snapshot().fact("bag", "position").source is Source.COSMOS
    ledger.commit([fact("S01-T1", source=Source.YOLO, label="under the table, left side")], writer="slow_loop")
    assert ledger.snapshot().fact("bag", "position").source is Source.YOLO


def test_description_trusts_cosmos_over_yolo(ledger):
    ledger.commit([take("S01-T1", True),
                   fact("S01-T1", field="description", source=Source.COSMOS, label="black backpack")],
                  writer="slow_loop")
    ledger.commit([fact("S01-T1", field="description", source=Source.YOLO, label="backpack")],
                  writer="slow_loop")
    assert ledger.snapshot().fact("bag", "description").label == "black backpack"


def test_facts_of_a_replaced_take_are_dropped(ledger):
    ledger.commit([take("S01-T1", True), fact("S01-T1", label="under the table, left side")], writer="slow_loop")
    ledger.commit([take("S01-T2", True)], writer="slow_loop")           # the new take brought no position fact
    assert ledger.snapshot().fact("bag", "position") is None            # the old take's fact must not linger


def test_facts_survive_a_failed_retake(ledger):
    ledger.commit([take("S01-T1", True), fact("S01-T1")], writer="slow_loop")
    ledger.commit([take("S01-T2", False)], writer="slow_loop")
    assert ledger.snapshot().fact("bag", "position").take_id == "S01-T1"


def test_reloading_the_plan_clears_progress(ledger):
    ledger.commit([take("S01-T1", True), fact("S01-T1")], writer="slow_loop")
    ledger.commit([PlanLoaded(plan=hitchcock_bag_under_table())], writer="planner")
    state = ledger.snapshot()
    assert state.coverage() == (0, 3) and state.scene == ()


# ───────────── Write log and persistence ─────────────

def test_every_write_is_logged_with_writer_and_sources(ledger):
    ledger.commit([take("S01-T1", True), fact("S01-T1")], writer="slow_loop")
    newest_first = ledger.recent(10)
    assert [e["kind"] for e in newest_first] == ["scene_fact_observed", "take_settled", "plan_loaded"]
    assert newest_first[0]["writer"] == "slow_loop" and newest_first[0]["sources"] == ["yolo"]
    assert newest_first[2]["writer"] == "planner" and newest_first[2]["sources"] == ["plan"]
    assert newest_first[0]["version"] == newest_first[1]["version"] == 2      # one commit, one version number


def test_state_survives_a_restart(tmp_path):
    path = tmp_path / "ledger.json"
    first = Ledger(path)
    first.commit([PlanLoaded(plan=hitchcock_bag_under_table())], writer="planner")
    first.commit([take("S01-T1", True), fact("S01-T1")], writer="slow_loop")

    second = Ledger(path)
    assert second.snapshot() == first.snapshot()
    assert second.snapshot().fact("bag", "position").label == "under the table, left side"
    assert len(second.recent(10)) == 3

    lines = (tmp_path / "ledger.events.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["event"]["fact"]["take_id"] == "S01-T1"   # the log holds the full event and can be replayed


def test_unreadable_file_is_set_aside_not_overwritten(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{ this is not a valid ledger", encoding="utf-8")
    book = Ledger(path)
    assert book.snapshot().shots == ()
    assert [p.name for p in tmp_path.iterdir() if ".bad-" in p.name], "the old file should have been moved aside and kept"
