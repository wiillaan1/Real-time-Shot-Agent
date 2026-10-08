"""The hand-written shot list (data, not logic).

This is the built-in example: three shots, all around the design doc's single example,
"the bag under the table". Each shot takes a different path, so together they exercise every branch of the loop:

    S01  produces a scene fact: after it passes, the ledger remembers where the bag is relative to the table
    S02  consumes that fact: guidance is derived from the ledger (bag on the left -> person on the right);
         its four constraints are the table in section 2.3 of the design doc
    S03  capability fallback: the pitch needs a gyro, a laptop webcam has none, so that constraint degrades to a text prompt

Constraints state relations (how far apart, not facing whom). There are no coordinates anywhere.
In fix / prompt, {a} {b} {ruler} become display names; {away} {ratio} etc. come from the predicate,
see each predicate in constraints.py.

Dependencies: contracts only.
"""
from __future__ import annotations

from .contracts import Checker, Constraint, Establish, Loop, Plan, Shot

SUBJECTS = {"person": "person", "bag": "bag", "table": "table"}

SETUP_A = "Setup A (table, front, medium shot)"
SETUP_B = "Setup B (high angle)"


def hitchcock_bag_under_table() -> Plan:
    s01 = Shot(
        id="S01",
        order=1,
        title="The bag under the table",
        description="Establishing shot: the table and the bag under it are both in frame, nobody is there yet.",
        intent="The audience learns about the bag before the character does: that gap is the suspense.",
        technique="Suspense: give the audience the information first",
        setup=SETUP_A,
        duration_s=5.0,
        constraints=(
            Constraint(
                id="S01.c1", text="The bag is in frame",
                checker=Checker.YOLO, loop=Loop.FAST,
                predicate="in_frame", args={"a": "bag"},
                fix="Put the {a} in frame: there is no {a} in the picture",
            ),
            Constraint(
                id="S01.c2", text="The bag is under the table",
                checker=Checker.YOLO, loop=Loop.FAST,
                predicate="under", args={"a": "bag", "b": "table"},
                fix="Put the {a} under the {b}, or lower the camera so the space under the table is in frame",
            ),
            Constraint(
                id="S01.c3", text="A viewer can see at a glance that there is a bag under the table",
                checker=Checker.COSMOS, loop=Loop.SLOW,
                ask="Is a bag clearly visible under the table in this clip?",
            ),
        ),
        establishes=(Establish(subject="bag", anchor="table"),),
    )

    s02 = Shot(
        id="S02",
        order=2,
        title="The person who has no idea",
        description="The person enters and sits at the table, busy with something: far from the bag, not looking its way.",
        intent="The character has no idea the bag is there, and the audience worries for them.",
        technique="Suspense: the character is unaware",
        setup=SETUP_A,
        duration_s=5.0,
        constraints=(
            Constraint(
                id="S02.c1", text="The person is in frame",
                checker=Checker.YOLO, loop=Loop.FAST,
                predicate="in_frame", args={"a": "person"},
                fix="Get the {a} in frame: there is no {a} in the picture",
            ),
            Constraint(
                id="S02.c2", text="Person and bag are more than half a table width apart",
                checker=Checker.YOLO, loop=Loop.FAST,
                predicate="apart",
                args={"a": "person", "b": "bag", "ruler": "table", "min_ratio": 0.5},
                fix="Move the {a} toward {away}: too close to the {b} ({ratio} table widths now, needs more than {min_ratio})",
            ),
            Constraint(
                id="S02.c3", text="The person is not facing the bag",
                checker=Checker.POSE, loop=Loop.FAST,
                predicate="not_facing", args={"a": "person", "b": "bag"},
                fix="Have the {a} turn the other way or face the camera: they are facing the {b}'s side ({toward})",
                prompt="Confirm the {a} is not looking toward the {b} (no pose estimation right now, cannot be checked automatically)",
            ),
            Constraint(
                id="S02.c4", text="A viewer can tell the person and the bag are in the same space",
                checker=Checker.COSMOS, loop=Loop.SLOW,
                ask="Can a viewer tell that the person and the bag are in the same space?",
            ),
        ),
    )

    s03 = Shot(
        id="S03",
        order=3,
        title="High-angle closing shot",
        description="Move up high and shoot down at the table and the person, who still has not noticed.",
        intent="From above the character looks small and exposed: a threatening close.",
        technique="High-angle shot",
        setup=SETUP_B,
        duration_s=5.0,
        constraints=(
            Constraint(
                id="S03.c1", text="The person is in frame",
                checker=Checker.YOLO, loop=Loop.FAST,
                predicate="in_frame", args={"a": "person"},
                fix="Get the {a} in frame: there is no {a} in the picture",
            ),
            Constraint(
                id="S03.c2", text="Camera pitch ≥ 25°",
                checker=Checker.SENSOR, loop=Loop.FAST, needs=("gyro",),
                predicate="pitch_at_least", args={"min_deg": 25},
                fix="Raise the camera and tilt it further down: pitch is only {deg}°, needs ≥ {min_deg}°",
                prompt="Shoot down from a high position (this video source has no gyro, so the pitch cannot be checked automatically: confirm it yourself)",
            ),
            Constraint(
                id="S03.c3", text="The picture is clearly a high-angle view",
                checker=Checker.COSMOS, loop=Loop.SLOW,
                ask="Is this clip shot from a clearly high angle looking down at the person?",
            ),
        ),
    )

    return Plan(style="Hitchcock-style suspense: the bag under the table", subjects=dict(SUBJECTS), shots=(s01, s02, s03))
