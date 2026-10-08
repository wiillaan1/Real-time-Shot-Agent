"""Shared test tools.

Rig = a running server + a simulated video source, driven by the tests the way a user would:
register the source, send frames, roll, end the take, wait for the check. Every request goes through the real HTTP routes.
"""
from __future__ import annotations

import json
import time
from contextlib import ExitStack
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from shotagent import media
from shotagent.config import Settings
from shotagent.contracts import Frame
from shotagent.judge.mock import MockJudge
from shotagent.perception.synthetic import SimScene, render
from shotagent.server import create_app
from shotagent.wiring import Services, build

SYNTHETIC = {"kind": "synthetic", "label": "Simulated", "width": 640, "height": 360, "fps": 3, "sensors": ["gyro"]}
WEBCAM = {"kind": "webcam", "label": "Test camera", "width": 640, "height": 360, "fps": 3, "sensors": []}

# A few pictures used all over
EMPTY_TABLE = SimScene(bag=None)
BAG_LEFT = SimScene(bag=0.2)
BAG_RIGHT = SimScene(bag=0.8)
PERSON_FAR = SimScene(bag=0.2, person=0.7, facing="right")      # the ideal S02 picture: far from the bag, back to it
PERSON_NEAR = SimScene(bag=0.2, person=0.4, facing="right")     # too close to the bag
PERSON_FACING_BAG = SimScene(bag=0.2, person=0.7, facing="left")


def jpeg_of(scene: SimScene) -> bytes:
    return media.encode_jpeg(render(scene), quality=90)


def frame_of(scene: SimScene, seq: int = 1, sensors: Optional[dict[str, float]] = None) -> Frame:
    """Build a frame directly, without HTTP (for unit tests of the fast / slow loop)."""
    data = jpeg_of(scene)
    now = time.time()
    return Frame(seq=seq, ts=now, received_at=now, jpeg=data, image=media.decode_jpeg(data),
                 sensors=dict(sensors or {}))


class Rig:
    def __init__(self, client: TestClient, services: Services):
        self.client = client
        self.services = services

    def register(self, caps: dict[str, Any] = SYNTHETIC) -> dict[str, Any]:
        reply = self.client.post("/api/source", json=caps)
        assert reply.status_code == 200, reply.text
        return reply.json()

    def push(self, scene: SimScene, sensors: Optional[dict[str, float]] = None) -> dict[str, Any]:
        headers = {"X-Frame-Ts": str(time.time()), "Content-Type": "image/jpeg"}
        if sensors:
            headers["X-Sensors"] = json.dumps(sensors)
        reply = self.client.post("/api/frame", content=jpeg_of(scene), headers=headers)
        assert reply.status_code == 200, reply.text
        return reply.json()

    def hint(self, scene: SimScene, sensors: Optional[dict[str, float]] = None) -> str:
        return self.push(scene, sensors)["guidance"]["headline"]

    def checks(self, scene: SimScene, sensors: Optional[dict[str, float]] = None) -> dict[str, dict]:
        return {c["constraint_id"]: c for c in self.push(scene, sensors)["guidance"]["checks"]}

    def state(self) -> dict[str, Any]:
        return self.client.get("/api/state").json()

    def post(self, path: str, **kwargs: Any):
        return self.client.post(path, **kwargs)

    def wait_until_checked(self, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.state()
            if state["phase"] != "checking":
                return state
            time.sleep(0.02)
        raise AssertionError("stuck in checking")

    def shoot(self, scene: SimScene, sensors: Optional[dict[str, float]] = None,
              frames: int = 6) -> dict[str, Any]:
        """Roll -> send a few frames -> end by hand -> wait for the check; returns the full state afterwards."""
        assert self.post("/api/shot/start").status_code == 200
        for _ in range(frames):
            self.push(scene, sensors)
        assert self.post("/api/shot/stop").status_code == 200
        return self.wait_until_checked()

    def shot(self, shot_id: str) -> dict[str, Any]:
        return next(r for r in self.state()["ledger"]["shots"] if r["shot"]["id"] == shot_id)

    def fact(self, subject: str, field: str) -> Optional[dict[str, Any]]:
        return next((f for f in self.state()["ledger"]["scene"]
                     if f["subject"] == subject and f["field"] == field), None)


@pytest.fixture
def make_rig(tmp_path):
    """Build a Rig. Accepts your own judge / detector / generate (shot-list generator) and any Settings field."""
    with ExitStack() as stack:
        def factory(judge=None, detector=None, register: Optional[dict] = SYNTHETIC, generate=None,
                    **overrides) -> Rig:
            overrides.setdefault("data_dir", tmp_path / "data")
            overrides.setdefault("detector", "none")
            settings = Settings(**overrides)
            services = build(settings, judge=judge or MockJudge(), detector=detector, generate=generate)
            client = stack.enter_context(TestClient(create_app(services=services)))
            rig = Rig(client, services)
            if register is not None:
                rig.register(register)
            return rig

        yield factory


@pytest.fixture
def rig(make_rig) -> Rig:
    return make_rig()
