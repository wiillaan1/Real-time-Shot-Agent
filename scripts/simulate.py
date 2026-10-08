"""Walk the whole agent loop in the terminal, explaining each step: no browser, no camera, no model.

    python scripts/simulate.py

It starts the server in-process and drives it through the HTTP endpoints only, as the browser does:
register a source -> send frames -> roll -> end the take -> wait for the check.
The pictures are simulated scenes (perception/synthetic.py), the semantic judge is the mock, and the ledger lives in a temp directory, not data/.
Run it after changing loop logic to see quickly whether the flow still works; the last line says whether it got through.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", message=".*httpx2.*")   # newer starlette's notice about its test client, unrelated to this script

from fastapi.testclient import TestClient  # noqa: E402

from shotagent import media  # noqa: E402
from shotagent.config import Settings  # noqa: E402
from shotagent.judge.mock import MockJudge  # noqa: E402
from shotagent.perception.synthetic import SimScene, render  # noqa: E402
from shotagent.server import create_app  # noqa: E402
from shotagent.wiring import build  # noqa: E402

SOURCE = {"kind": "synthetic", "label": "Simulated", "width": 640, "height": 360, "fps": 0, "sensors": ["gyro"]}


class Walkthrough:
    def __init__(self, client: TestClient):
        self.client = client

    def say(self, text: str = "") -> None:
        print(text)

    def push(self, scene: SimScene, pitch: float | None = None) -> dict:
        headers = {"X-Frame-Ts": str(time.time())}
        if pitch is not None:
            headers["X-Sensors"] = json.dumps({"pitch": pitch})
        reply = self.client.post("/api/frame", content=media.encode_jpeg(render(scene), 90), headers=headers)
        reply.raise_for_status()
        return reply.json()

    def look(self, what: str, scene: SimScene, pitch: float | None = None) -> None:
        """Framing: send one frame and see what the fast loop says."""
        guidance = self.push(scene, pitch)["guidance"]
        marks = {"ok": "✓", "violated": "✕", "unknown": "?", "degraded": "-", "pending": "..."}
        checks = " ".join(marks[c["state"]] for c in guidance["checks"])
        self.say(f"  Picture: {what}")
        self.say(f"    fast loop -> {guidance['headline']}   [{checks}]")
        for check in guidance["checks"]:
            if "ledger" in check["basis"].values():
                self.say(f"             (in {check['constraint_id']} the hidden subject was recovered from the ledger)")
        for note in guidance["notes"]:
            self.say(f"             note: {note}")

    def shoot(self, scene: SimScene, pitch: float | None = None, frames: int = 8) -> dict:
        """Shoot one take: roll -> send frames -> end -> wait for the slow loop to settle it."""
        self.client.post("/api/shot/start").raise_for_status()
        for _ in range(frames):
            self.push(scene, pitch)
        self.client.post("/api/shot/stop").raise_for_status()
        while True:
            state = self.client.get("/api/state").json()
            if state["phase"] != "checking":
                break
            time.sleep(0.05)
        self.say(f"  Shot one take ({frames} frames)")
        self.say(f"    slow loop -> {state['notice']['text']}")
        return state

    def ledger(self, state: dict) -> None:
        status = {"todo": "to do", "passed": "passed", "retake": "retake"}
        names = state["ledger"]["subjects"]
        shots = "  ".join(f"{r['shot']['id']} {status[r['status']]}" for r in state["ledger"]["shots"])
        self.say(f"    ledger v{state['ledger']['version']}: {shots}")
        for fact in state["ledger"]["scene"]:
            if fact["field"] == "position":
                self.say(f"      scene state: {names.get(fact['subject'], fact['subject'])} {fact['label']}"
                         f" (source {fact['source']}, from {fact['take_id']}, confidence {fact['confidence']})")
        for line in state["briefing"]:
            self.say(f"      guidance for the next shot: {line}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        judge = MockJudge(script={"S02": [False, True]})      # fail S02's first take, to walk through a reshoot
        services = build(Settings(data_dir=Path(tmp), detector="none"), judge=judge)
        with TestClient(create_app(services=services)) as client:
            w = Walkthrough(client)
            client.post("/api/source", json=SOURCE).raise_for_status()

            w.say("Shot one, S01 \"The bag under the table\": get the bag's position relative to the table into the ledger")
            w.look("just the table", SimScene(bag=None))
            w.look("bag on top of the table", SimScene(bag=0.2, bag_under=False))
            w.look("bag under the table, on the left", SimScene(bag=0.2))
            w.ledger(w.shoot(SimScene(bag=0.2)))

            w.say()
            w.say("Shot two, S02 \"The person who has no idea\": guidance derived from the ledger")
            w.look("person sitting next to the bag", SimScene(bag=0.2, person=0.4, facing="right"))
            w.look("person on the right, but facing the bag", SimScene(bag=0.2, person=0.7, facing="left"))
            w.look("person on the right, back to the bag", SimScene(bag=0.2, person=0.7, facing="right"))
            w.look("bag hidden from view", SimScene(bag=None, person=0.7, facing="right"))
            w.say("  (the mock is set to fail the first take)")
            state = w.shoot(SimScene(bag=0.2, person=0.7, facing="right"))
            w.say(f"    still on {state['current_shot_id']}, no advance")
            w.ledger(w.shoot(SimScene(bag=0.2, person=0.7, facing="right")))

            w.say()
            w.say("Shot three, S03 \"High-angle closing shot\": the pitch comes from the source's gyro")
            w.look("pitch 10°", SimScene(bag=0.2, person=0.5), pitch=10)
            w.look("pitch 35°", SimScene(bag=0.2, person=0.5), pitch=35)
            state = w.shoot(SimScene(bag=0.2, person=0.5), pitch=35)
            w.ledger(state)

            w.say()
            w.say("Move the bag to the right and reshoot shot one: shot two's guidance follows")
            client.post("/api/shot/select", json={"shot_id": "S01"}).raise_for_status()
            w.shoot(SimScene(bag=0.8))
            client.post("/api/shot/select", json={"shot_id": "S02"}).raise_for_status()
            flipped = client.get("/api/state").json()
            w.ledger(flipped)
            w.look("person still sitting on the right", SimScene(bag=0.8, person=0.7, facing="left"))

            finished = state["phase"] == "done" and state["coverage"] == {"passed": 3, "total": 3}
            guidance_flipped = any("on the left side of the table" in line for line in flipped["briefing"])
            w.say()
            w.say(f"All three shots passed: {'yes' if finished else 'no'}; "
                  f"guidance flipped after the bag changed sides: {'yes' if guidance_flipped else 'no'}")
            return 0 if finished and guidance_flipped else 1


if __name__ == "__main__":
    sys.exit(main())
