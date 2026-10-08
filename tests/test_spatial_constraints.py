"""Relative-position geometry and relation predicates, tested with hand-written boxes, no images involved."""
from __future__ import annotations

import pytest

from shotagent.constraints import LEDGER, LIVE, World, brief, evaluate
from shotagent.contracts import Box, CheckState, Detection, Observation, Source
from shotagent.ledger.state import SceneFact
from shotagent.plans import SUBJECTS, hitchcock_bag_under_table
from shotagent.spatial import facing_from_keypoints, project, relative_position

PLAN = hitchcock_bag_under_table()
CONSTRAINTS = {c.id: c for shot in PLAN.shots for c in shot.constraints}

TABLE = Box(x1=0.25, y1=0.52, x2=0.75, y2=0.92)


def bag_at(u: float, under: bool = True) -> Box:
    cx = TABLE.x1 + u * TABLE.w
    y2 = 0.92 if under else 0.52
    return Box(x1=cx - 0.045, y1=y2 - 0.12, x2=cx + 0.045, y2=y2)


def person_at(x: float) -> Box:
    return Box(x1=x - 0.06, y1=0.15, x2=x + 0.06, y2=0.75)


def keypoints(x: float, facing: str) -> dict:
    """Keypoints of a head in a 16:9 picture: the nose left / right of the head's centre, or dead centre."""
    shift = {"left": -0.03, "right": 0.03, "front": 0.0}[facing]
    points = {
        "nose": (x + shift, 0.22, 0.9),
        "left_shoulder": (x - 0.06, 0.30, 0.9),
        "right_shoulder": (x + 0.06, 0.30, 0.9),
    }
    if facing == "front":
        points["left_ear"] = (x - 0.035, 0.22, 0.9)
        points["right_ear"] = (x + 0.035, 0.22, 0.9)
    else:
        points["left_ear"] = (x, 0.22, 0.9)
    return points


def world(*detections: Detection, memory=None, sensors=None) -> World:
    observation = Observation(seq=1, ts=0, detections=detections, sensors=sensors or {})
    return World(observation, memory or {})


def det(label: str, box: Box, **extra) -> Detection:
    return Detection(label=label, box=box, **extra)


def remembered_bag(u: float) -> dict[str, SceneFact]:
    rel = relative_position(bag_at(u), TABLE)
    return {"bag": SceneFact(
        subject="bag", field="position", value=rel.as_value("table"),
        label=f"under the table, {'left side' if u < 0.5 else 'right side'}", source=Source.YOLO, confidence=0.9, take_id="S01-T1",
    )}


# ───────────── Relative position ─────────────

def test_relative_position_uses_the_anchor_as_ruler():
    rel = relative_position(bag_at(0.2), TABLE)
    assert rel.u == pytest.approx(0.2) and rel.side == "left" and rel.vertical == "under"
    assert relative_position(bag_at(0.8), TABLE).side == "right"
    assert relative_position(bag_at(0.5), TABLE).side == "center"
    assert relative_position(bag_at(0.2, under=False), TABLE).vertical == "above"


def test_relative_position_is_scale_free():
    """The same scene shot twice as close (all boxes scaled and shifted) has the same relative position: that is the point of not using pixels."""
    def zoom(b: Box) -> Box:
        return Box(x1=b.x1 * 1.6 - 0.3, y1=b.y1 * 1.6 - 0.5, x2=b.x2 * 1.6 - 0.3, y2=b.y2 * 1.6 - 0.5)

    near, far = relative_position(zoom(bag_at(0.2)), zoom(TABLE)), relative_position(bag_at(0.2), TABLE)
    assert near.u == pytest.approx(far.u) and near.v == pytest.approx(far.v)
    assert near.side == far.side and near.vertical == far.vertical


def test_project_inverts_relative_position():
    """The fast loop can restore the relative position the slow loop wrote to the ledger from the table as seen now, even if the table moved in the picture."""
    value = relative_position(bag_at(0.2), TABLE).as_value("table")
    moved_table = Box(x1=0.40, y1=0.40, x2=0.80, y2=0.72)
    restored = project(value, moved_table)
    again = relative_position(restored, moved_table)
    assert again.u == pytest.approx(0.2, abs=1e-3) and again.side == "left" and again.vertical == "under"


# ───────────── Facing ─────────────

@pytest.mark.parametrize("facing", ["left", "right", "front"])
def test_facing_from_keypoints(facing):
    assert facing_from_keypoints(keypoints(0.5, facing)) == facing


def test_facing_is_unknown_without_usable_keypoints():
    assert facing_from_keypoints(None) is None
    low_confidence = {name: (x, y, 0.05) for name, (x, y, _) in keypoints(0.5, "left").items()}
    assert facing_from_keypoints(low_confidence) is None


def test_two_ears_without_a_nose_reads_as_back_turned():
    points = keypoints(0.5, "front")
    del points["nose"]
    assert facing_from_keypoints(points) == "back"


def face(nose: float, eyes=None, ears=None) -> dict:
    """Keypoints of a face. eyes / ears are ((x, confidence), (x, confidence)): the person's left, then right."""
    points = {"nose": (nose, 0.30, 0.99)}
    for part, pair in (("eye", eyes), ("ear", ears)):
        for side, (x, conf) in zip(("left", "right"), pair or ()):
            points[f"{side}_{part}"] = (x, 0.28, conf)
    return points


def test_facing_falls_back_to_the_eyes_when_hair_hides_the_ears():
    """Hair over the ears (neither is clear): the two eyes still tell frontal from profile, instead of getting stuck on "can't tell"."""
    hidden = ((0.54, 0.10), (0.46, 0.08))
    assert facing_from_keypoints(face(0.500, eyes=((0.52, 0.95), (0.48, 0.97)), ears=hidden)) == "front"
    assert facing_from_keypoints(face(0.506, eyes=((0.52, 0.95), (0.48, 0.97)), ears=hidden)) == "front"   # turned just slightly
    assert facing_from_keypoints(face(0.485, eyes=((0.52, 0.95), (0.48, 0.97)), ears=hidden)) == "left"
    assert facing_from_keypoints(face(0.515, eyes=((0.52, 0.95), (0.48, 0.97)), ears=hidden)) == "right"


def test_a_frontal_face_with_one_hidden_ear_still_reads_as_front():
    """Facing the camera with one ear unclear (hair, a hand): one remaining ear must not be read as a profile."""
    one_ear = ((0.54, 0.96), (0.46, 0.20))
    assert facing_from_keypoints(face(0.502, eyes=((0.52, 0.99), (0.48, 0.98)), ears=one_ear)) == "front"


def test_a_shaky_far_side_point_is_not_trusted_as_half_of_a_pair():
    """In profile the far eye is only the model's guess (low confidence, wrong place): it is not paired with the near eye; the one clear ear decides."""
    profile_left = face(0.44, eyes=((0.47, 0.98), (0.40, 0.42)), ears=((0.52, 0.97), (0.50, 0.02)))
    assert facing_from_keypoints(profile_left) == "left"
    profile_right = face(0.56, eyes=((0.60, 0.40), (0.53, 0.97)), ears=((0.50, 0.03), (0.48, 0.96)))
    assert facing_from_keypoints(profile_right) == "right"


def test_facing_does_not_depend_on_scale_or_position():
    near = face(0.485, eyes=((0.52, 0.95), (0.48, 0.97)))
    far = {name: (0.8 + (x - 0.5) * 0.3, y, conf) for name, (x, y, conf) in near.items()}    # the person smaller and moved to frame right
    assert facing_from_keypoints(near) == facing_from_keypoints(far) == "left"


# ───────────── Predicates ─────────────

def test_apart_measures_in_table_widths_and_says_which_way_to_move():
    c = CONSTRAINTS["S02.c2"]
    near = evaluate(c, world(det("person", person_at(0.4)), det("bag", bag_at(0.2)), det("table", TABLE)), SUBJECTS)
    assert near.state is CheckState.VIOLATED
    assert near.measured["ratio"] == pytest.approx(0.1, abs=0.01)
    assert "frame right" in near.message and "0.10" in near.message

    far = evaluate(c, world(det("person", person_at(0.7)), det("bag", bag_at(0.2)), det("table", TABLE)), SUBJECTS)
    assert far.state is CheckState.OK and far.message == ""


def test_apart_flips_direction_when_the_bag_is_on_the_other_side():
    """Demo moment two of the design doc: with the bag on the right, the guidance turns to the left by itself. The constraint did not change by a character."""
    c = CONSTRAINTS["S02.c2"]
    result = evaluate(c, world(det("person", person_at(0.6)), det("bag", bag_at(0.8)), det("table", TABLE)), SUBJECTS)
    assert result.state is CheckState.VIOLATED and "frame left" in result.message


def test_apart_falls_back_to_the_ledger_when_the_bag_is_not_visible():
    c = CONSTRAINTS["S02.c2"]
    seen = world(det("person", person_at(0.4)), det("table", TABLE), memory=remembered_bag(0.2))
    result = evaluate(c, seen, SUBJECTS)
    assert result.state is CheckState.VIOLATED
    assert result.basis == {"person": LIVE, "table": LIVE, "bag": LEDGER}
    assert result.measured["ratio"] == pytest.approx(0.1, abs=0.01)     # the same as measured when the bag is visible


def test_live_detection_wins_over_the_ledger():
    c = CONSTRAINTS["S02.c2"]
    seen = world(det("person", person_at(0.4)), det("bag", bag_at(0.9)), det("table", TABLE),
                 memory=remembered_bag(0.2))
    result = evaluate(c, seen, SUBJECTS)
    assert result.basis["bag"] == LIVE and result.state is CheckState.OK


def test_apart_cannot_judge_without_its_ruler_or_any_idea_where_the_bag_is():
    c = CONSTRAINTS["S02.c2"]
    no_table = evaluate(c, world(det("person", person_at(0.4)), det("bag", bag_at(0.2))), SUBJECTS)
    assert no_table.state is CheckState.UNKNOWN and "table" in no_table.message
    no_bag = evaluate(c, world(det("person", person_at(0.4)), det("table", TABLE)), SUBJECTS)
    assert no_bag.state is CheckState.UNKNOWN and "bag" in no_bag.message


def test_in_frame_only_counts_what_is_actually_visible():
    c = CONSTRAINTS["S01.c1"]
    assert evaluate(c, world(det("bag", bag_at(0.2))), SUBJECTS).state is CheckState.OK
    remembered_only = evaluate(c, world(det("table", TABLE), memory=remembered_bag(0.2)), SUBJECTS)
    assert remembered_only.state is CheckState.VIOLATED       # in the ledger is not the same as in the frame


def test_under():
    c = CONSTRAINTS["S01.c2"]
    assert evaluate(c, world(det("bag", bag_at(0.2)), det("table", TABLE)), SUBJECTS).state is CheckState.OK
    on_top = evaluate(c, world(det("bag", bag_at(0.2, under=False)), det("table", TABLE)), SUBJECTS)
    assert on_top.state is CheckState.VIOLATED
    assert evaluate(c, world(det("bag", bag_at(0.2))), SUBJECTS).state is CheckState.UNKNOWN


def test_not_facing():
    c = CONSTRAINTS["S02.c3"]

    def check(facing: str) -> CheckState:
        person = det("person", person_at(0.7), keypoints=keypoints(0.7, facing))
        return evaluate(c, world(person, det("bag", bag_at(0.2)), det("table", TABLE)), SUBJECTS).state

    assert check("left") is CheckState.VIOLATED       # the bag is to the person's left and the person faces left
    assert check("right") is CheckState.OK
    assert check("front") is CheckState.OK
    no_pose = evaluate(c, world(det("person", person_at(0.7)), det("bag", bag_at(0.2))), SUBJECTS)
    assert no_pose.state is CheckState.UNKNOWN


def test_pitch_reads_the_sensor():
    c = CONSTRAINTS["S03.c2"]
    assert evaluate(c, world(sensors={"pitch": 35.0}), SUBJECTS).state is CheckState.OK
    low = evaluate(c, world(sensors={"pitch": 10.0}), SUBJECTS)
    assert low.state is CheckState.VIOLATED and "10°" in low.message
    assert evaluate(c, world(), SUBJECTS).state is CheckState.UNKNOWN


# ───────────── Guidance derived from the ledger ─────────────

def test_briefing_is_derived_from_the_ledger_and_flips_with_the_bag():
    left = brief(CONSTRAINTS["S02.c2"], remembered_bag(0.2), SUBJECTS)
    right = brief(CONSTRAINTS["S02.c2"], remembered_bag(0.8), SUBJECTS)
    assert "keep the person on the right side of the table" in left and "S01-T1" in left and "YOLO" in left     # where to go, and on what basis
    assert "keep the person on the left side of the table" in right
    facing = brief(CONSTRAINTS["S02.c3"], remembered_bag(0.2), SUBJECTS)
    assert facing == "The bag is on the left side of the table: the person should not look that way"
    assert brief(CONSTRAINTS["S02.c2"], {}, SUBJECTS) is None             # nothing in the ledger, nothing made up
