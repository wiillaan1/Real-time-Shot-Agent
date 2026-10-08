"""The whole agent loop, driven through the HTTP endpoints only, as the browser does, with simulated pictures.

These tests are the product behaviours of the design doc: pass before leaving the setup, one kept take per shot,
shot two's guidance derived from shot one, text prompts when the source lacks a capability, resuming after a restart.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from conftest import (
    BAG_LEFT, BAG_RIGHT, EMPTY_TABLE, PERSON_FAR, PERSON_NEAR, WEBCAM, Rig, frame_of, jpeg_of,
)

from shotagent.contracts import JudgeRequest, JudgeVerdict
from shotagent.judge.base import JudgeError
from shotagent.judge.mock import MockJudge
from shotagent.perception.synthetic import SimScene, SyntheticDetector

HIGH_ANGLE = SimScene(bag=0.2, person=0.5)
PITCH = {"pitch": 35.0}


def run_all_three(rig: Rig) -> dict:
    rig.shoot(BAG_LEFT)
    rig.shoot(PERSON_FAR)
    return rig.shoot(HIGH_ANGLE, PITCH)


# ───────────── Main flow ─────────────

def test_fresh_start(rig):
    state = rig.state()
    assert state["phase"] == "framing" and state["current_shot_id"] == "S01"
    assert state["coverage"] == {"passed": 0, "total": 3}
    assert [r["status"] for r in state["ledger"]["shots"]] == ["todo", "todo", "todo"]
    assert state["runtime"]["detector"] == "synthetic" and state["runtime"]["judge"] == "mock"


def test_full_run_through_three_shots(rig):
    after_s01 = rig.shoot(BAG_LEFT)
    assert after_s01["current_shot_id"] == "S02" and after_s01["coverage"]["passed"] == 1
    assert after_s01["notice"]["kind"] == "pass" and "Same setup" in after_s01["notice"]["text"]

    after_s02 = rig.shoot(PERSON_FAR)
    assert after_s02["current_shot_id"] == "S03"
    assert "You can move to Setup B (high angle) now" in after_s02["notice"]["text"]

    done = rig.shoot(HIGH_ANGLE, PITCH)
    assert done["phase"] == "done" and done["current_shot_id"] is None
    assert done["coverage"] == {"passed": 3, "total": 3}
    assert "All shots are done" in done["notice"]["text"]
    assert [r["effective_take"]["take_id"] for r in done["ledger"]["shots"]] == ["S01-T1", "S02-T1", "S03-T1"]


def test_shot_two_guidance_is_derived_from_shot_one(rig):
    """Demo moment two: shot one remembers the bag on the left -> shot two puts the person on the right;
    reshoot shot one with the bag on the right -> it flips to the left by itself."""
    assert rig.state()["briefing"] == []
    rig.shoot(BAG_LEFT)
    fact = rig.fact("bag", "position")
    assert (fact["label"], fact["source"], fact["take_id"]) == ("under the table, left side", "yolo", "S01-T1")
    assert "keep the person on the right side of the table" in rig.state()["briefing"][0]   # before the take: from the ledger, relative to the table
    assert "frame right" in rig.hint(PERSON_NEAR)                    # per frame: which way to move in the picture

    assert rig.post("/api/shot/select", json={"shot_id": "S01"}).json()["current_shot_id"] == "S01"
    after = rig.shoot(BAG_RIGHT)
    assert after["current_shot_id"] == "S02"
    assert rig.fact("bag", "position")["label"] == "under the table, right side"
    assert rig.fact("bag", "position")["take_id"] == "S01-T2"
    assert "keep the person on the left side of the table" in after["briefing"][0]
    assert "frame left" in rig.hint(SimScene(bag=0.8, person=0.6, facing="left"))


def test_ledger_stands_in_for_a_bag_that_left_the_frame(rig):
    rig.shoot(BAG_LEFT)
    guidance = rig.push(SimScene(bag=None, person=0.4, facing="right"))["guidance"]
    check = next(c for c in guidance["checks"] if c["constraint_id"] == "S02.c2")
    assert check["state"] == "violated" and check["basis"]["bag"] == "ledger"
    assert check["readout"] == "0.10 table widths"

    # The UI has to draw the bag recovered from the ledger: it sits where it was relative to the table in shot one
    (recalled,) = guidance["recalled"]
    assert recalled["label"] == "bag" and recalled["take_id"] == "S01-T1"
    center_x = (recalled["box"]["x1"] + recalled["box"]["x2"]) / 2
    assert abs(center_x - 0.35) < 0.02                       # table's left edge 0.25 + 0.2 x table width 0.5
    assert rig.push(PERSON_FAR)["guidance"]["recalled"] == []   # nothing to recover when the bag is visible


# ───────────── Product rules ─────────────

def test_failed_check_keeps_the_user_on_the_same_shot(make_rig):
    rig = make_rig(judge=MockJudge(script={"S02": [False, True]}))
    rig.shoot(BAG_LEFT)

    failed = rig.shoot(PERSON_FAR)
    assert failed["phase"] == "framing" and failed["current_shot_id"] == "S02"
    assert failed["notice"]["kind"] == "fail" and "Stay where you are and reshoot" in failed["notice"]["text"]
    assert rig.shot("S02")["status"] == "retake"
    assert rig.shot("S02")["effective_take"]["take_id"] == "S02-T1"       # the latest take is kept for now

    passed = rig.shoot(PERSON_FAR)
    assert passed["current_shot_id"] == "S03"
    assert rig.shot("S02")["status"] == "passed" and rig.shot("S02")["attempts"] == 2
    assert rig.shot("S02")["effective_take"]["take_id"] == "S02-T2"


def test_the_last_result_stays_on_screen_until_the_next_one_replaces_it(make_rig):
    rig = make_rig(judge=MockJudge(script={"S01": [False, True]}))
    assert rig.state()["notice"] is None                              # nothing shot yet: this row shows the current phase
    failed = rig.shoot(BAG_LEFT)
    assert failed["notice"]["kind"] == "fail" and failed["notice"]["take_id"] == "S01-T1"

    retaking = rig.post("/api/shot/start").json()
    assert retaking["phase"] == "recording"
    assert retaking["notice"]["take_id"] == "S01-T1"                  # while reshooting, why the last take failed is still visible
    for _ in range(4):
        rig.push(BAG_LEFT)
    rig.post("/api/shot/stop")
    passed = rig.wait_until_checked()
    assert passed["notice"]["kind"] == "pass" and passed["notice"]["take_id"] == "S01-T2"


def test_take_shot_badly_fails_on_the_fast_constraints(rig):
    rig.shoot(BAG_LEFT)
    asked_before = len(rig.services.judge.requests)
    failed = rig.shoot(PERSON_NEAR)
    assert failed["current_shot_id"] == "S02" and "table width" in failed["notice"]["text"]
    assert len(rig.services.judge.requests) == asked_before


def test_user_waits_in_place_while_the_take_is_checked(make_rig):
    rig = make_rig(judge=MockJudge(latency_s=0.4))
    rig.post("/api/shot/start")
    for _ in range(4):
        rig.push(BAG_LEFT)
    during = rig.post("/api/shot/stop").json()
    assert during["phase"] == "checking" and "stay where you are" in during["banner"]
    assert during["checking"] == {"take_id": "S01-T1", "shot_id": "S01", "frames": 4}

    reply = rig.push(BAG_LEFT)
    assert reply["phase"] == "checking" and reply["guidance"] is None
    assert rig.post("/api/shot/start").status_code == 409
    assert rig.post("/api/reset").status_code == 409

    after = rig.wait_until_checked()
    assert after["phase"] == "framing" and after["current_shot_id"] == "S02" and after["checking"] is None


def test_reshooting_a_passed_shot_and_failing_keeps_the_old_take(rig):
    run_all_three(rig)
    rig.post("/api/shot/select", json={"shot_id": "S01"})
    rig.post("/api/debug/mock-judge", json={"next": "fail"})
    after = rig.shoot(BAG_RIGHT)

    assert after["notice"]["kind"] == "fail" and "is kept" in after["notice"]["text"]
    assert rig.shot("S01")["status"] == "passed"
    assert rig.shot("S01")["effective_take"]["take_id"] == "S01-T1"
    assert rig.fact("bag", "position")["label"] == "under the table, left side"   # the failed take did not change the scene state either
    assert after["phase"] == "done"                                       # a manual pick covers one take, then the automatic order resumes
    takes = sorted(p.name.split("_")[0] for p in (rig.services.settings.data_dir / "takes" / "S01").iterdir())
    assert takes == ["S01-T1", "S01-T2"]                                  # old and failed files are all still on disk
    assert rig.client.get("/api/takes/S01-T2/thumb").status_code == 404   # but not in the index


# ───────────── Start and end of a take ─────────────

def test_fixed_timer_ends_the_take(make_rig):
    rig = make_rig(take_seconds=0.4)
    started = rig.post("/api/shot/start").json()
    assert started["phase"] == "recording" and started["take"]["duration_s"] == 0.4
    deadline = time.time() + 5
    while rig.push(BAG_LEFT)["phase"] == "recording":
        assert time.time() < deadline, "the timer ran out but the take did not end"
        time.sleep(0.03)
    state = rig.wait_until_checked()
    assert state["current_shot_id"] == "S02"
    take_dir = Path(rig.shot("S01")["effective_take"]["clip_dir"])
    assert json.loads((take_dir / "take.json").read_text("utf-8"))["end_reason"] == "timer"


def test_only_frames_that_arrive_while_recording_join_the_take(rig):
    for _ in range(3):
        rig.push(EMPTY_TABLE)                    # frames while framing do not count
    rig.post("/api/shot/start")
    for _ in range(4):
        reply = rig.push(BAG_LEFT)
    assert reply["take"]["frames"] == 4 and reply["take"]["take_id"] == "S01-T1"
    rig.post("/api/shot/stop")
    rig.wait_until_checked()
    rig.push(BAG_LEFT)
    assert rig.shot("S01")["effective_take"]["n_frames"] == 4


def test_take_that_is_too_short_is_not_sent_for_checking(rig):
    rig.post("/api/shot/start")
    rig.push(BAG_LEFT)
    after = rig.post("/api/shot/stop").json()
    assert after["phase"] == "framing" and after["notice"]["kind"] == "error" and "too short" in after["notice"]["text"]
    assert rig.shot("S01")["attempts"] == 0 and rig.services.judge.requests == []


def test_actions_are_refused_in_the_wrong_phase(rig):
    assert rig.post("/api/shot/stop").status_code == 409
    assert rig.post("/api/shot/start").status_code == 200
    assert rig.post("/api/shot/start").status_code == 409
    assert rig.post("/api/shot/select", json={"shot_id": "S02"}).status_code == 409
    assert rig.post("/api/shot/select", json={"shot_id": "S02"}).json()["error"]


def test_selecting_an_unknown_shot_is_refused(rig):
    assert rig.post("/api/shot/select", json={"shot_id": "S99"}).status_code == 404


def test_a_manual_reshoot_can_be_called_off(rig):
    run_all_three(rig)
    picked = rig.post("/api/shot/select", json={"shot_id": "S01"}).json()
    assert picked["phase"] == "framing" and picked["current_shot_id"] == "S01" and picked["manual_selection"]

    back = rig.post("/api/shot/select", json={"shot_id": None}).json()       # "Cancel reshoot" in the UI
    assert back["phase"] == "done" and back["current_shot_id"] is None and not back["manual_selection"]
    assert rig.shot("S01")["effective_take"]["take_id"] == "S01-T1" and rig.shot("S01")["attempts"] == 1


# ───────────── Video source and degrading ─────────────

def test_frames_need_a_registered_source(make_rig):
    rig = make_rig(register=None)
    reply = rig.client.post("/api/frame", content=jpeg_of(BAG_LEFT))
    assert reply.status_code == 409 and reply.json() == {"error": "no_source"}   # the client registers again on this
    rig.register()
    assert rig.push(BAG_LEFT)["dropped"] is False
    assert rig.client.post("/api/frame", content=b"not an image").status_code == 400
    assert rig.client.post("/api/frame", content=b"").status_code == 400     # browsers may send an empty frame before the camera has a picture
    assert rig.push(BAG_LEFT)["guidance"]["ready"]                           # back to normal after a bad frame


def test_registration_tells_the_source_how_to_upload(make_rig):
    rig = make_rig(register=None, upload_fps=4.0)
    assert rig.register()["upload"] == {"fps": 4.0, "width": 640, "jpeg_quality": 0.7}


def test_webcam_without_a_detector_still_completes_the_loop_on_text_prompts(make_rig):
    """--detector none: before YOLO is installed, text prompts + the semantic judge still get through the list."""
    rig = make_rig(register=WEBCAM)
    state = rig.state()
    assert state["runtime"]["detector"] == "none" and state["runtime"]["capabilities"] == []
    assert {m["mode"] for m in state["modes"].values()} == {"degraded", "slow"}
    guidance = rig.push(BAG_LEFT)["guidance"]
    assert guidance["ready"] and guidance["level"] == "info" and "self-check" in guidance["headline"]
    prompts = [c["message"] for c in guidance["checks"] if c["state"] == "degraded"]
    assert prompts == ["Please confirm: The bag is in frame (cannot be checked automatically, please confirm yourself)",
                       "Please confirm: The bag is under the table (cannot be checked automatically, please confirm yourself)"]

    done = run_all_three(rig)
    assert done["phase"] == "done"
    verdicts = rig.shot("S01")["effective_take"]["verdicts"]
    assert [v["passed"] for v in verdicts] == [None, None, True]          # unchecked is recorded as unchecked, never as passed
    assert rig.fact("bag", "position") is None                            # no detection, no position fact


def test_webcam_with_a_detector_degrades_only_what_the_source_cannot_measure(make_rig):
    rig = make_rig(register=WEBCAM, detector=SyntheticDetector())
    rig.shoot(BAG_LEFT)
    rig.shoot(PERSON_FAR)
    modes = rig.state()["modes"]
    assert modes["S03.c1"]["mode"] == "check"
    assert modes["S03.c2"]["mode"] == "degraded" and "gyro" in modes["S03.c2"]["note"]
    reply = rig.push(HIGH_ANGLE, {"pitch": 80.0})
    assert reply["guidance"]["observation"]["sensors"] == {}             # readings from a source that declared no sensor are ignored
    assert {c["constraint_id"]: c["state"] for c in reply["guidance"]["checks"]}["S03.c2"] == "degraded"
    done = rig.shoot(HIGH_ANGLE)
    assert done["phase"] == "done"
    by_id = {v["constraint_id"]: v for v in rig.shot("S03")["effective_take"]["verdicts"]}
    assert by_id["S03.c2"]["passed"] is None and by_id["S03.c2"]["source"] == "sensor"


def test_sensor_readings_travel_with_the_frame_when_the_source_declares_the_sensor(rig):
    assert rig.push(BAG_LEFT, {"pitch": 35.0})["guidance"]["observation"]["sensors"] == {"pitch": 35.0}


def test_switching_source_mid_take_discards_the_take(rig):
    rig.post("/api/shot/start")
    rig.push(BAG_LEFT)
    after = rig.register(WEBCAM)["state"]
    assert after["phase"] == "framing" and after["take"] is None and "void" in after["notice"]["text"]
    assert rig.shot("S01")["attempts"] == 0


def test_sim_endpoint_renders_what_the_detector_reads(rig):
    reply = rig.client.get("/api/sim/frame", params={"bag": 0.8, "person": 0.3, "facing": "left"})
    assert reply.status_code == 200 and reply.headers["content-type"] == "image/jpeg"
    pushed = rig.client.post("/api/frame", content=reply.content).json()
    assert {d["label"] for d in pushed["guidance"]["observation"]["detections"]} == {"table", "bag", "person"}
    empty = rig.client.get("/api/sim/frame", params={"table": "false"})
    assert rig.client.post("/api/frame", content=empty.content).json()["guidance"]["observation"]["detections"] == []


# ───────────── Errors, restart, starting over ─────────────

def test_judge_failure_is_shown_and_changes_nothing(make_rig):
    class Unreachable:
        name = "unreachable"

        def judge(self, request: JudgeRequest) -> JudgeVerdict:
            raise JudgeError("Cannot reach the Cosmos endpoint")

    rig = make_rig(judge=Unreachable())
    version = rig.state()["ledger"]["version"]
    after = rig.shoot(BAG_LEFT)
    assert after["phase"] == "framing" and after["current_shot_id"] == "S01"
    assert after["notice"]["kind"] == "error" and "Cannot reach the Cosmos endpoint" in after["notice"]["text"]
    assert after["ledger"]["version"] == version and rig.shot("S01")["attempts"] == 0
    assert rig.post("/api/shot/start").status_code == 200                # the state machine is not stuck: shooting can go on


def test_restart_resumes_where_it_stopped(make_rig, tmp_path):
    first = make_rig(data_dir=tmp_path / "shared")
    first.shoot(BAG_LEFT)

    second = make_rig(data_dir=tmp_path / "shared")
    state = second.state()
    assert state["current_shot_id"] == "S02" and state["coverage"]["passed"] == 1
    assert "right side of the table" in state["briefing"][0]             # the scene state is back too
    assert [e["kind"] for e in state["log"]][-1] == "plan_loaded"         # and so is the write log
    assert second.client.get("/api/takes/S01-T1/thumb").status_code == 200


def test_reset_starts_over(rig):
    rig.shoot(BAG_LEFT)
    state = rig.post("/api/reset").json()
    assert state["current_shot_id"] == "S01" and state["coverage"]["passed"] == 0
    assert state["ledger"]["scene"] == [] and state["notice"] is None and state["briefing"] == []


def test_mock_verdict_switch_only_accepts_pass_or_fail(rig):
    assert rig.post("/api/debug/mock-judge", json={"next": "fail"}).json()["runtime"]["judge_forced"] is False
    assert rig.post("/api/debug/mock-judge", json={"next": None}).json()["runtime"]["judge_forced"] is None
    assert rig.post("/api/debug/mock-judge", json={"next": "maybe"}).status_code == 422


def test_low_confidence_fact_is_flagged_for_confirmation(rig):
    rig.shoot(SimScene(bag=0.5))
    assert rig.state()["confirm"] == ["bag.position"]
    rig.post("/api/shot/select", json={"shot_id": "S01"})
    rig.shoot(BAG_LEFT)
    assert rig.state()["confirm"] == []


def test_thumbnail_of_an_indexed_take(rig):
    rig.shoot(BAG_LEFT)
    reply = rig.client.get("/api/takes/S01-T1/thumb")
    assert reply.status_code == 200 and reply.headers["content-type"] == "image/jpeg" and len(reply.content) > 1000
    assert rig.client.get("/api/takes/..%2F..%2Fledger/thumb").status_code == 404


def test_rev_only_moves_when_there_is_something_new_to_fetch(rig):
    first = rig.push(BAG_LEFT)["rev"]
    assert rig.push(BAG_LEFT)["rev"] == first            # a frame arriving is no reason for the UI to refetch the state
    assert rig.post("/api/shot/start").json()["rev"] > first


# ───────────── Only the latest frame is processed ─────────────

def test_a_frame_arriving_while_another_is_being_processed_is_dropped(make_rig):
    class Slow(SyntheticDetector):
        def detect(self, image):
            time.sleep(0.2)
            return super().detect(image)

    rig = make_rig(register=WEBCAM, detector=Slow())
    session = rig.services.session

    async def two_at_once():
        return await asyncio.gather(
            session.on_frame(frame_of(BAG_LEFT, seq=1)),
            session.on_frame(frame_of(BAG_LEFT, seq=2)),
        )

    first, second = asyncio.run(two_at_once())
    assert (first["dropped"], second["dropped"]) == (False, True)
    assert first["guidance"]["ready"] and second["guidance"] is None


def test_a_frame_that_arrived_before_the_take_started_does_not_join_it(make_rig):
    """The frame arrived while framing; by the time detection finished a take had started. It does not belong to that take."""
    class Slow(SyntheticDetector):
        def detect(self, image):
            time.sleep(0.2)
            return super().detect(image)

    rig = make_rig(register=WEBCAM, detector=Slow())
    session = rig.services.session

    async def start_while_a_frame_is_in_flight():
        in_flight = asyncio.create_task(session.on_frame(frame_of(BAG_LEFT, seq=1)))
        await asyncio.sleep(0.05)
        session.start_take()
        early = await in_flight
        during = await session.on_frame(frame_of(BAG_LEFT, seq=2))
        session._discard_take()          # tidy up: do not leave the timer running
        return early, during

    early, during = asyncio.run(start_while_a_frame_is_in_flight())
    assert early["phase"] == "recording" and early["take"]["frames"] == 0
    assert during["take"]["frames"] == 1
