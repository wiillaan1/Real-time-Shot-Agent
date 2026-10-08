"""The real YOLO detector (optional): runs only when ultralytics is installed and the weights can be fetched, else skipped.

The first run downloads yolo11n.pt and yolo11n-pose.pt (about 6 MB each).
The picture is one of ultralytics' own sample images (a bus and a few pedestrians).
It verifies the adapter: class mapping, normalised boxes, keypoint names. It does not verify whether your bag is recognised:
check that against the real scene with scripts/check_yolo.py.
"""
from __future__ import annotations

import time

import cv2
import pytest

pytest.importorskip("ultralytics")

from ultralytics.utils import ASSETS  # noqa: E402

from shotagent.config import DEFAULT_LABEL_MAP  # noqa: E402
from shotagent.contracts import CAP_DETECT, CAP_POSE, CheckState, Frame, SourceCaps  # noqa: E402
from shotagent.fast_loop import FastLoop  # noqa: E402
from shotagent.ledger.store import Ledger  # noqa: E402
from shotagent.perception.yolo import COCO_KEYPOINTS, YoloDetector  # noqa: E402
from shotagent.planner import Planner  # noqa: E402
from shotagent.spatial import DEFAULT_TUNING, facing_from_keypoints  # noqa: E402


@pytest.fixture(scope="module")
def detector():
    try:
        return YoloDetector(label_map=DEFAULT_LABEL_MAP)
    except Exception as exc:   # e.g. offline, weights cannot be downloaded
        pytest.skip(f"YOLO weights could not be loaded: {exc}")


@pytest.fixture(scope="module")
def street():
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    assert image is not None
    return image


def test_reports_both_capabilities(detector):
    assert detector.capabilities == {CAP_DETECT, CAP_POSE}


def test_people_come_with_normalised_boxes_and_named_keypoints(detector, street):
    people = [d for d in detector.detect(street) if d.label == "person"]
    assert len(people) >= 3
    for person in people:
        box = person.box
        assert 0 <= box.x1 < box.x2 <= 1 and 0 <= box.y1 < box.y2 <= 1
        assert set(person.keypoints) == set(COCO_KEYPOINTS)
        assert all(0 <= conf <= 1 for _, _, conf in person.keypoints.values())
    clearest = max(people, key=lambda d: d.conf)
    assert facing_from_keypoints(clearest.keypoints) in ("left", "right", "front", "back")


def test_only_mapped_roles_are_reported(detector, street):
    assert {d.label for d in detector.detect(street)} <= set(DEFAULT_LABEL_MAP.values())   # the bus is not one of the roles


def test_fast_loop_runs_on_a_real_detector(detector, street):
    ledger = Ledger()
    Planner(ledger).load()
    fast = FastLoop(lambda caps: detector, DEFAULT_TUNING)
    fast.load(ledger.snapshot(), "S02", SourceCaps(kind="webcam"))
    now = time.time()
    guidance = fast.step(Frame(seq=1, ts=now, received_at=now, jpeg=b"", image=street, sensors={}))
    states = {c.constraint_id: c.state for c in guidance.checks}
    assert states["S02.c1"] is CheckState.OK              # the person is in frame
    assert states["S02.c2"] is CheckState.UNKNOWN         # no table in the street, so distance cannot be measured
    assert "table" in guidance.headline
