"""Fast loop: reading the snapshot, degrading by capability, giving hints."""
from __future__ import annotations

from typing import Optional

from conftest import BAG_LEFT, EMPTY_TABLE, PERSON_FACING_BAG, PERSON_FAR, PERSON_NEAR, frame_of

from shotagent import texts
from shotagent.contracts import CAP_DETECT, CheckState, Detection, Source, SourceCaps
from shotagent.fast_loop import FastLoop
from shotagent.ledger.events import PlanLoaded, SceneFactObserved, TakeSettled
from shotagent.ledger.state import LedgerState, SceneFact, TakeRecord
from shotagent.ledger.store import Ledger
from shotagent.perception.null import NullDetector
from shotagent.perception.synthetic import SimScene, SyntheticDetector
from shotagent.plans import hitchcock_bag_under_table
from shotagent.spatial import DEFAULT_TUNING

SIM = SourceCaps(kind="synthetic", sensors=("gyro",))
SIM_NO_GYRO = SourceCaps(kind="synthetic")


class BoxesOnly:
    """A detector with boxes but no pose (as if only the YOLO detection model were loaded)."""

    name, note = "boxes-only", ""
    capabilities = frozenset({CAP_DETECT})

    def detect(self, image) -> list[Detection]:
        return [d.model_copy(update={"keypoints": None}) for d in SyntheticDetector().detect(image)]


def snapshot(bag_u: Optional[float] = None) -> LedgerState:
    """A ledger snapshot; with bag_u given it is as if S01 had been shot and the bag's position recorded."""
    ledger = Ledger()
    ledger.commit([PlanLoaded(plan=hitchcock_bag_under_table())], writer="planner")
    if bag_u is not None:
        side = "left" if bag_u < 0.5 else "right"
        ledger.commit([
            TakeSettled(take=TakeRecord(take_id="S01-T1", shot_id="S01", started_at=0, ended_at=5,
                                        n_frames=10, clip_dir="/clips", passed=True)),
            SceneFactObserved(fact=SceneFact(
                subject="bag", field="position",
                value={"anchor": "table", "u": bag_u, "v": 0.85, "sw": 0.18, "sh": 0.3,
                       "side": side, "vertical": "under"},
                label=f"under the table, {'left side' if side == 'left' else 'right side'}",
                source=Source.YOLO, confidence=0.9, take_id="S01-T1")),
        ], writer="slow_loop")
    return ledger.snapshot()


def loop(detector=None) -> FastLoop:
    detector = detector or SyntheticDetector()
    return FastLoop(lambda caps: detector, DEFAULT_TUNING)


def states(guidance) -> dict[str, CheckState]:
    return {c.constraint_id: c.state for c in guidance.checks}


# ───────────── Hints ─────────────

def test_guides_step_by_step_until_ready():
    fast = loop()
    fast.load(snapshot(), "S01", SIM)

    first = fast.step(frame_of(EMPTY_TABLE))
    assert not first.ready and first.level == "fix" and "bag" in first.headline

    on_table = fast.step(frame_of(SimScene(bag=0.2, bag_under=False)))
    assert not on_table.ready and "under" in on_table.headline

    good = fast.step(frame_of(BAG_LEFT))
    assert good.ready and good.level == "ok" and good.headline == texts.READY
    assert states(good) == {"S01.c1": CheckState.OK, "S01.c2": CheckState.OK, "S01.c3": CheckState.PENDING}


def test_observation_travels_with_the_guidance():
    """What the fast loop saw travels out with the hint: that is how the slow loop later writes YOLO's positions to the ledger."""
    fast = loop()
    fast.load(snapshot(), "S01", SIM)
    guidance = fast.step(frame_of(BAG_LEFT, seq=7))
    assert guidance.seq == 7 and guidance.observation.seq == 7
    assert {d.label for d in guidance.observation.detections} == {"table", "bag"}


def test_headline_follows_plan_order():
    fast = loop()
    fast.load(snapshot(bag_u=0.2), "S02", SIM)
    assert fast.step(frame_of(PERSON_NEAR)).headline.startswith("Move the person toward frame right")      # "too close" first
    assert fast.step(frame_of(PERSON_FACING_BAG)).headline.startswith("Have the person turn the other way")   # then the facing
    assert fast.step(frame_of(PERSON_FAR)).ready


# ───────────── Ledger snapshot ─────────────

def test_briefing_comes_from_the_snapshot():
    fast = loop()
    fast.load(snapshot(), "S02", SIM)
    assert fast.describe()["briefing"] == []                 # S01 not shot yet: the ledger has no position for the bag

    fast.load(snapshot(bag_u=0.2), "S02", SIM)
    briefing = " ".join(fast.describe()["briefing"])
    assert "right side of the table" in briefing and "S01-T1" in briefing

    fast.load(snapshot(bag_u=0.8), "S02", SIM)
    assert "left side of the table" in " ".join(fast.describe()["briefing"])


def test_uses_the_ledger_for_a_bag_it_cannot_see():
    fast = loop()
    fast.load(snapshot(bag_u=0.2), "S02", SIM)
    hidden_bag_near = fast.step(frame_of(SimScene(bag=None, person=0.4, facing="right")))
    check = next(c for c in hidden_bag_near.checks if c.constraint_id == "S02.c2")
    assert check.state is CheckState.VIOLATED and check.basis["bag"] == "ledger"

    hidden_bag_far = fast.step(frame_of(SimScene(bag=None, person=0.7, facing="right")))
    assert hidden_bag_far.ready


def test_without_a_ledger_entry_a_hidden_bag_cannot_be_judged():
    fast = loop()
    fast.load(snapshot(), "S02", SIM)
    guidance = fast.step(frame_of(SimScene(bag=None, person=0.7, facing="right")))
    assert states(guidance)["S02.c2"] is CheckState.UNKNOWN and not guidance.ready


def test_snapshot_is_only_refreshed_by_load():
    """The ledger changing during a take does not affect the fast loop; it sees the new state at the next load."""
    fast = loop()
    fast.load(snapshot(), "S02", SIM)
    version_before = fast.describe()["ledger_version"]
    newer = snapshot(bag_u=0.2)                      # the ledger has moved on
    assert fast.describe()["ledger_version"] == version_before and fast.describe()["briefing"] == []
    fast.load(newer, "S02", SIM)
    assert fast.describe()["ledger_version"] == newer.version and fast.describe()["briefing"]


def test_says_so_when_the_frame_disagrees_with_the_ledger():
    fast = loop()
    fast.load(snapshot(bag_u=0.2), "S02", SIM)
    moved = fast.step(frame_of(SimScene(bag=0.8, person=0.3, facing="left")))
    assert len(moved.notes) == 1 and "right side" in moved.notes[0] and "left side" in moved.notes[0]
    assert fast.step(frame_of(PERSON_FAR)).notes == ()


def test_no_disagreement_note_while_reshooting_the_shot_that_establishes_the_fact():
    fast = loop()
    fast.load(snapshot(bag_u=0.2), "S01", SIM)
    assert fast.step(frame_of(SimScene(bag=0.8))).notes == ()


# ───────────── Degrading by capability ─────────────

def test_sensor_constraint_degrades_on_a_source_without_the_sensor():
    fast = loop()
    fast.load(snapshot(), "S03", SIM_NO_GYRO)
    modes = fast.describe()["modes"]
    assert modes["S03.c2"]["mode"] == "degraded" and "gyro" in modes["S03.c2"]["note"]
    assert modes["S03.c1"]["mode"] == "check"

    guidance = fast.step(frame_of(SimScene(person=0.5), sensors={"pitch": 5.0}))
    check = next(c for c in guidance.checks if c.constraint_id == "S03.c2")
    assert check.state is CheckState.DEGRADED and "high position" in check.message
    assert guidance.ready                 # a constraint that cannot be checked does not block the take, it only gets a text prompt
    assert guidance.headline == texts.READY_SELF_CHECK     # but it does not claim "all satisfied" either: one item is left to the user


def test_sensor_constraint_is_checked_when_the_source_has_the_sensor():
    fast = loop()
    fast.load(snapshot(), "S03", SIM)
    assert fast.describe()["modes"]["S03.c2"]["mode"] == "check"
    low = fast.step(frame_of(SimScene(person=0.5), sensors={"pitch": 5.0}))
    assert not low.ready and "pitch is only 5°" in low.headline
    assert fast.step(frame_of(SimScene(person=0.5), sensors={"pitch": 40.0})).ready


def test_pose_constraint_degrades_without_a_pose_model():
    fast = loop(BoxesOnly())
    fast.load(snapshot(bag_u=0.2), "S02", SIM)
    modes = fast.describe()["modes"]
    assert modes["S02.c3"]["mode"] == "degraded" and "pose" in modes["S02.c3"]["note"]
    assert modes["S02.c2"]["mode"] == "check"
    facing_bag = fast.step(frame_of(PERSON_FACING_BAG))
    assert facing_bag.ready                                    # facing cannot be checked, distance still is
    assert states(facing_bag)["S02.c3"] is CheckState.DEGRADED
    assert facing_bag.headline == texts.READY_SELF_CHECK and facing_bag.level == "ok"
    too_close = fast.step(frame_of(PERSON_NEAR))
    assert not too_close.ready and "too close" in too_close.headline   # a checkable one failed: say that first


def test_everything_degrades_to_text_without_a_detector():
    fast = loop(NullDetector())
    fast.load(snapshot(bag_u=0.2), "S02", SourceCaps(kind="webcam"))
    guidance = fast.step(frame_of(PERSON_NEAR))
    assert states(guidance) == {
        "S02.c1": CheckState.DEGRADED, "S02.c2": CheckState.DEGRADED,
        "S02.c3": CheckState.DEGRADED, "S02.c4": CheckState.PENDING,
    }
    assert guidance.level == "info" and guidance.ready and guidance.headline == texts.SELF_CHECK_ONLY
    prompts = [c.message for c in guidance.checks if c.state is CheckState.DEGRADED]
    assert len(prompts) == 3 and all("confirm" in prompt.lower() for prompt in prompts)   # one text prompt per degraded constraint
    assert fast.describe()["briefing"], "guidance derived from the ledger needs no detector and must still be given"


def test_detector_is_chosen_from_the_source_description():
    camera, sim = NullDetector(), SyntheticDetector()
    fast = FastLoop(lambda caps: sim if caps and caps.kind == "synthetic" else camera, DEFAULT_TUNING)
    fast.load(snapshot(), "S01", SourceCaps(kind="webcam"))
    assert fast.describe()["detector"] == "none"
    fast.load(snapshot(), "S01", SIM)
    assert fast.describe()["detector"] == "synthetic"


def test_idle_when_there_is_no_shot():
    fast = loop()
    fast.load(snapshot(), None, SIM)
    guidance = fast.step(frame_of(BAG_LEFT))
    assert guidance.shot_id is None and not guidance.ready and guidance.checks == ()
