"""Switching shot lists: from submitting an idea / a script in the UI to the new list in the ledger and shot to the end. All over HTTP.

The generator is fake (it returns a prepared list); the real step is tested in test_plan_gen.py.
"""
from __future__ import annotations

import asyncio
import threading

from conftest import BAG_LEFT, PERSON_FAR, WEBCAM
from test_plan_gen import GOOD, ROLES

from shotagent.contracts import Box, Detection
from shotagent.perception.synthetic import SimScene
from shotagent.plan_gen import PlanError, to_plan

IDEA = "Film someone who forgot their bag under the table"
HIGH_ANGLE = SimScene(bag=0.2, person=0.5)


class FakeGenerator:
    """Same shape as PlanGenerator: (idea, allowed roles) -> Plan."""

    name = "fake-model"

    def __init__(self, fail: str = ""):
        self.fail = fail
        self.calls: list[tuple[str, object]] = []

    def __call__(self, request, roles=None):
        self.calls.append((request, roles))
        if self.fail:
            raise PlanError(self.fail)
        plan, errors = to_plan(GOOD, request, ROLES, 6)
        assert not errors
        return plan


def test_a_new_script_replaces_the_shot_list_and_clears_progress(make_rig):
    generate = FakeGenerator()
    rig = make_rig(generate=generate)
    rig.shoot(BAG_LEFT)                                       # one shot of the example list is already done
    assert rig.state()["coverage"]["passed"] == 1 and rig.state()["runtime"]["plan_model"] == "fake-model"

    reply = rig.post("/api/plan", json={"request": IDEA})
    state = reply.json()
    assert reply.status_code == 200 and generate.calls[0][0] == IDEA
    assert (state["ledger"]["style"], state["ledger"]["request"]) == ("The bag left under the table", IDEA)
    assert state["ledger"]["synopsis"] and state["ledger"]["subjects"]["bag"] == "backpack"
    assert state["coverage"] == {"passed": 0, "total": 3} and state["ledger"]["scene"] == []
    assert (state["phase"], state["current_shot_id"], state["planning"]) == ("framing", "S01", None)
    assert state["notice"]["kind"] == "info" and "The bag left under the table" in state["notice"]["text"]
    assert state["log"][0]["writer"] == "planner"            # switching lists also goes through the planner's one write path


def test_a_generated_list_can_be_shot_to_the_end(make_rig):
    rig = make_rig(generate=FakeGenerator())
    rig.post("/api/plan", json={"request": IDEA})

    assert rig.hint(SimScene(bag=None)) == "Get the backpack in frame: there is no backpack in the picture"   # hints use the story's names
    rig.shoot(BAG_LEFT)
    assert rig.fact("bag", "position")["label"] == "under the long table, left side"     # remember -> scene fact
    assert any("keep the owner on the right side of the long table" in line for line in rig.state()["briefing"])  # the next shot's guidance comes from it
    assert "too close to the backpack" in rig.hint(SimScene(bag=0.2, person=0.3, facing="right"))
    rig.shoot(PERSON_FAR)
    state = rig.shoot(HIGH_ANGLE, {"pitch": 35.0})
    assert state["phase"] == "done" and state["coverage"] == {"passed": 3, "total": 3}


def test_the_generator_is_told_what_the_current_detector_can_see(make_rig):
    in_sim = FakeGenerator()
    make_rig(generate=in_sim).post("/api/plan", json={"request": IDEA})
    assert in_sim.calls[0][1] == frozenset({"person", "bag", "table"})           # the simulated picture can only draw these three

    on_camera = FakeGenerator()
    make_rig(generate=on_camera, register=WEBCAM).post("/api/plan", json={"request": IDEA})
    assert on_camera.calls[0][1] is None                                        # no detector: no limit


def test_a_failed_generation_changes_nothing(make_rig):
    rig = make_rig(generate=FakeGenerator(fail="The model's shot list failed validation"))
    rig.shoot(BAG_LEFT)
    before = rig.state()

    reply = rig.post("/api/plan", json={"request": IDEA})
    after = rig.state()
    assert reply.status_code == 502 and "The model's shot list failed validation" in reply.json()["error"]
    assert after["ledger"] == before["ledger"]                                  # the ledger did not change by a character, progress is intact
    assert (after["phase"], after["current_shot_id"]) == ("framing", "S02")


def test_without_a_model_only_the_example_list_is_available(rig):
    assert rig.state()["runtime"]["plan_model"] is None
    reply = rig.post("/api/plan", json={"request": IDEA})
    assert reply.status_code == 502 and "OPENAI_API_KEY" in reply.json()["error"]
    assert rig.state()["ledger"]["request"] == ""


def test_an_empty_request_goes_back_to_the_example_list(make_rig):
    generate = FakeGenerator()
    rig = make_rig(generate=generate)
    rig.post("/api/plan", json={"request": IDEA})
    state = rig.post("/api/plan", json={"request": "  "}).json()
    assert state["ledger"]["request"] == "" and state["ledger"]["subjects"]["bag"] == "bag"
    assert len(generate.calls) == 1                                             # going back to the example needs no model


def test_starting_over_keeps_the_generated_script_without_asking_the_model_again(make_rig):
    generate = FakeGenerator()
    rig = make_rig(generate=generate)
    rig.post("/api/plan", json={"request": IDEA})
    rig.shoot(BAG_LEFT)

    state = rig.post("/api/reset").json()
    assert state["ledger"]["style"] == "The bag left under the table" and state["ledger"]["request"] == IDEA
    assert state["coverage"]["passed"] == 0 and len(generate.calls) == 1


def test_a_restart_keeps_the_generated_script_and_its_progress(make_rig, tmp_path):
    first = make_rig(generate=FakeGenerator(), data_dir=tmp_path / "shared")
    first.post("/api/plan", json={"request": IDEA})
    first.shoot(BAG_LEFT)

    second = make_rig(data_dir=tmp_path / "shared")                             # restart, this time with no model configured at all
    state = second.state()
    assert state["ledger"]["style"] == "The bag left under the table" and state["notice"] is None
    assert state["coverage"]["passed"] == 1 and state["current_shot_id"] == "S02"


def test_the_script_cannot_be_changed_in_the_middle_of_a_take(make_rig):
    generate = FakeGenerator()
    rig = make_rig(generate=generate)
    rig.post("/api/shot/start")
    assert rig.post("/api/plan", json={"request": IDEA}).status_code == 409 and not generate.calls
    rig.post("/api/shot/stop")


def test_an_overlong_request_is_refused(make_rig):
    assert make_rig(generate=FakeGenerator()).post("/api/plan", json={"request": "x" * 4001}).status_code == 422


def test_nothing_can_be_shot_while_the_list_is_being_written(make_rig):
    """The model takes ten seconds or more: meanwhile the session sits in "planning": no hints, no take, no second submission."""
    release = threading.Event()

    class Slow(FakeGenerator):
        def __call__(self, request, roles=None):
            release.wait(5)
            return super().__call__(request, roles)

    rig = make_rig(generate=Slow())
    session = rig.services.session

    async def scenario():
        writing = asyncio.create_task(session.replan(IDEA))
        await asyncio.sleep(0.05)
        during = session.view()
        refused = []
        for action in (session.start_take, session.reset):
            try:
                action()
            except Exception as exc:
                refused.append(type(exc).__name__)
        try:
            await session.replan("another idea")
        except Exception as exc:
            refused.append(type(exc).__name__)
        release.set()
        await writing
        return during, refused

    during, refused = asyncio.run(scenario())
    assert during["phase"] == "planning" and during["planning"] == {"request": IDEA, "model": "fake-model"}
    assert refused == ["SessionError"] * 3
    assert rig.state()["phase"] == "framing" and rig.state()["ledger"]["style"] == "The bag left under the table"


def test_detections_that_are_not_in_the_script_are_left_out(make_rig):
    """The detector knows chairs, cups... but this list has none of them: they are not passed on, and no extra boxes appear."""
    class SeesAChair:
        name, note, capabilities, roles = "stub", "", frozenset({"detect"}), frozenset({"person", "chair"})

        def detect(self, image):
            box = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.5)
            return [Detection(label="person", box=box), Detection(label="chair", box=box)]

    rig = make_rig(register=WEBCAM, detector=SeesAChair())
    labels = [d["label"] for d in rig.push(BAG_LEFT)["guidance"]["observation"]["detections"]]
    assert labels == ["person"]
