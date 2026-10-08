"""Slow loop: settling a take and writing the conclusions to the ledger."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import BAG_LEFT, PERSON_FAR, PERSON_NEAR, frame_of

from shotagent.contracts import JudgeRequest, JudgeVerdict, ShotStatus, Source, SourceCaps, Take
from shotagent.fast_loop import FastLoop
from shotagent.judge.base import JudgeError
from shotagent.judge.mock import MockJudge
from shotagent.ledger.store import Ledger
from shotagent.perception.synthetic import SimScene, SyntheticDetector
from shotagent.planner import Planner
from shotagent.slow_loop import SlowLoop
from shotagent.spatial import DEFAULT_TUNING
from shotagent.takes import TakeBuffer, TakeStore

SIM = SourceCaps(kind="synthetic", sensors=("gyro",))


class Stage:
    """Ledger + both loops, without the session or HTTP: shoot a set of pictures into a take by hand, then settle it."""

    def __init__(self, tmp_path, judge=None):
        self.ledger = Ledger(tmp_path / "ledger.json")
        Planner(self.ledger).load()
        self.judge = judge or MockJudge()
        self.fast = FastLoop(lambda caps: SyntheticDetector(), DEFAULT_TUNING)
        self.slow = SlowLoop(self.ledger, self.judge, DEFAULT_TUNING)
        self.store = TakeStore(tmp_path / "takes")

    def record(self, shot_id: str, scenes: list[SimScene], take_no: int = 1) -> Take:
        state = self.ledger.snapshot()
        self.fast.load(state, shot_id, SIM)
        buffer = TakeBuffer(f"{shot_id}-T{take_no}", state.record(shot_id).shot)
        for index, scene in enumerate(scenes):
            frame = frame_of(scene, seq=index + 1)
            buffer.add(frame, self.fast.step(frame))
        return self.store.save(buffer, ended_at=buffer.started_at + 5.0, end_reason="manual")

    def settle(self, shot_id: str, scenes: list[SimScene], take_no: int = 1):
        return self.slow.settle(self.record(shot_id, scenes, take_no))


@pytest.fixture
def stage(tmp_path) -> Stage:
    return Stage(tmp_path)


def test_passing_take_writes_shot_log_and_scene_state_in_one_commit(stage):
    before = stage.ledger.snapshot().version
    result = stage.settle("S01", [BAG_LEFT] * 6)
    state = stage.ledger.snapshot()

    assert result.passed and result.shot_status is ShotStatus.PASSED
    assert state.version == before + 1                         # the take's verdict and the scene facts are one commit
    record = state.record("S01")
    assert record.effective_take.take_id == "S01-T1" and record.effective_take.n_frames == 6
    assert record.effective_take.judge == "mock"               # the ledger honestly records that the mock judged this one

    position = state.fact("bag", "position")
    assert position.source is Source.YOLO and position.take_id == "S01-T1"
    assert position.value["anchor"] == "table" and position.value["side"] == "left"
    assert position.value["u"] == pytest.approx(0.2, abs=0.02)
    assert position.label == "under the table, left side"
    assert state.fact("bag", "description").source is Source.COSMOS


def test_each_verdict_is_tagged_with_who_checked_it(stage):
    stage.settle("S01", [BAG_LEFT] * 6)
    stage.settle("S02", [PERSON_FAR] * 6)
    result = stage.settle("S03", [SimScene(person=0.5)] * 6)
    by_id = {v.constraint_id: v for v in result.record.verdicts}
    assert by_id["S03.c1"].source is Source.YOLO
    assert by_id["S03.c2"].source is Source.SENSOR             # no pitch reading -> cannot tell -> fails
    assert by_id["S03.c2"].passed is False
    sources = {v.constraint_id: v.source for v in stage.ledger.snapshot().record("S02").effective_take.verdicts}
    assert sources == {"S02.c1": Source.YOLO, "S02.c2": Source.YOLO, "S02.c3": Source.YOLO,
                       "S02.c4": Source.COSMOS}


def test_fast_constraint_failure_skips_the_judge_and_asks_for_a_retake(stage):
    stage.settle("S01", [BAG_LEFT] * 6)
    calls_before = len(stage.judge.requests)
    result = stage.settle("S02", [PERSON_NEAR] * 6)

    assert not result.passed and result.shot_status is ShotStatus.RETAKE
    assert len(stage.judge.requests) == calls_before, "a fast constraint failed, so no seconds should be spent asking the semantic model"
    by_id = {v.constraint_id: v for v in result.record.verdicts}
    assert by_id["S02.c2"].passed is False and by_id["S02.c2"].score == 0.0
    assert by_id["S02.c4"].passed is None                      # not sent for checking, which is not the same as failing
    assert any("table width" in reason for reason in result.reasons)


def test_fast_constraints_tolerate_some_bad_frames(stage):
    """By share of frames: 4 of 6 satisfying (67%) passes, 3 (50%) does not. The default threshold is 60%."""
    stage.settle("S01", [BAG_LEFT] * 6)
    half = stage.settle("S02", [PERSON_FAR] * 3 + [PERSON_NEAR] * 3, take_no=1)
    assert not half.passed
    mostly = stage.settle("S02", [PERSON_FAR] * 4 + [PERSON_NEAR] * 2, take_no=2)
    assert mostly.passed
    verdict = next(v for v in mostly.record.verdicts if v.constraint_id == "S02.c2")
    assert verdict.score == pytest.approx(0.67, abs=0.01) and "4/6" in verdict.detail


def test_judge_saying_no_fails_the_take_and_writes_no_facts(tmp_path):
    stage = Stage(tmp_path, judge=MockJudge(script={"S01": [False]}))
    result = stage.settle("S01", [BAG_LEFT] * 6)
    state = stage.ledger.snapshot()
    assert not result.passed and state.record("S01").status is ShotStatus.RETAKE
    assert state.record("S01").effective_take.take_id == "S01-T1"     # kept even though it failed: it is the latest take
    assert state.scene == ()                                           # but no fact may be taken from a failed take


def test_judge_is_asked_the_slow_constraints_about_this_clip(stage):
    take = stage.record("S02", [PERSON_FAR] * 6)
    stage.slow.settle(take)
    request: JudgeRequest = stage.judge.requests[-1]
    assert [q.constraint_id for q in request.questions] == ["S02.c4"]
    assert request.questions[0].ask.startswith("Can a viewer tell")
    assert request.clip_dir == take.dir and set(request.subjects) == {"person", "bag", "table"}


def test_judge_failure_leaves_the_ledger_untouched(tmp_path):
    class Broken:
        name = "broken"

        def judge(self, request: JudgeRequest) -> JudgeVerdict:
            raise JudgeError("unreachable")

    stage = Stage(tmp_path, judge=Broken())
    before = stage.ledger.snapshot()
    with pytest.raises(JudgeError):
        stage.settle("S01", [BAG_LEFT] * 6)
    assert stage.ledger.snapshot() == before


def test_position_fact_is_less_confident_when_left_or_right_is_unclear(stage):
    result = stage.settle("S01", [SimScene(bag=0.5)] * 6)
    fact = stage.ledger.snapshot().fact("bag", "position")
    assert result.passed and fact.value["side"] == "center"
    assert fact.confidence < DEFAULT_TUNING.confirm_below       # the UI will flag "please confirm"


def test_position_fact_only_counts_frames_where_both_are_visible(stage):
    scenes = [BAG_LEFT] * 4 + [SimScene(table=False, bag=0.2)] * 2     # the table is out of frame for two frames
    result = stage.settle("S01", scenes)
    fact = stage.ledger.snapshot().fact("bag", "position")
    assert result.passed and fact.confidence == pytest.approx(4 / 6, abs=0.01)


def test_reshooting_updates_the_fact_and_a_failed_reshoot_does_not(stage):
    stage.settle("S01", [BAG_LEFT] * 6, take_no=1)
    stage.judge.force_next(False)
    kept = stage.settle("S01", [SimScene(bag=0.8)] * 6, take_no=2)
    assert kept.kept_previous and kept.shot_status is ShotStatus.PASSED
    assert stage.ledger.snapshot().fact("bag", "position").value["side"] == "left"

    stage.settle("S01", [SimScene(bag=0.8)] * 6, take_no=3)
    fact = stage.ledger.snapshot().fact("bag", "position")
    assert fact.value["side"] == "right" and fact.take_id == "S01-T3"


def test_take_is_saved_with_frames_and_per_frame_results(stage, tmp_path):
    take = stage.record("S01", [BAG_LEFT] * 5)
    directory = Path(take.dir)
    assert directory.parent == tmp_path / "takes" / "S01"
    assert sorted(p.name for p in directory.iterdir()) == [
        "000.jpg", "001.jpg", "002.jpg", "003.jpg", "004.jpg", "take.json"]
    reloaded = Take.model_validate_json((directory / "take.json").read_text("utf-8"))
    assert reloaded == take and len(reloaded.frames[0].checks) == 3
