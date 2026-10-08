"""Wiring: every object in the system is created and connected here.

To see who holds whom, read build() and nothing else. Permission to write the ledger is also handed out here:

    the Ledger object goes to exactly two holders: Planner (writes the shot list) and SlowLoop (settles takes, writes scene facts)
    Planner    also gets a generator (idea / script -> list, an LLM behind it); there is none without a key
    FastLoop   gets no ledger object at all, only a snapshot when each shot starts
    Session    gets two read-only functions: ledger.snapshot and ledger.recent

Detectors are assigned by source type: SyntheticDetector for the simulated picture, the configured one (YOLO or null) for everything else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .config import ROLE_NAMES, Settings
from .contracts import SourceCaps
from .fast_loop import FastLoop
from .frame_source import FrameSource
from .judge import build_judge
from .judge.base import Judge
from .ledger.store import Ledger
from .llm import ChatLLM
from .perception import build_detector
from .perception.base import Detector
from .perception.synthetic import SyntheticDetector
from .plan_gen import PlanGenerator
from .planner import Generate, Planner
from .session import Session
from .slow_loop import SlowLoop
from .takes import TakeStore


@dataclass
class Services:
    settings: Settings
    ledger: Ledger
    planner: Planner
    judge: Judge
    camera_detector: Detector
    fast: FastLoop
    slow: SlowLoop
    takes: TakeStore
    frames: FrameSource
    session: Session


def build_generator(settings: Settings) -> Optional[PlanGenerator]:
    """The generator that turns an idea / script into a shot list. Without a key there is none and the planner only has the hand-written example."""
    if not settings.llm_api_key:
        return None
    llm = ChatLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_s)
    # Roles the model may use = every role the detector recognises (the right-hand side of the label map)
    roles = {role: ROLE_NAMES.get(role, role) for role in dict.fromkeys(settings.label_map.values())}
    return PlanGenerator(llm.complete, roles, max_shots=settings.plan_max_shots,
                         attempts=settings.plan_attempts, name=llm.model)


def build(settings: Settings, *, judge: Optional[Judge] = None, detector: Optional[Detector] = None,
          generate: Optional[Generate] = None) -> Services:
    """judge / detector / generate can be injected (for tests); otherwise they are built from the settings."""
    ledger = Ledger(settings.data_dir / "ledger.json")

    planner = Planner(ledger, generate or build_generator(settings))   # writes the ledger: shot list

    camera_detector = detector or build_detector(settings)
    synthetic_detector = SyntheticDetector()

    def pick_detector(caps: Optional[SourceCaps]) -> Detector:
        if caps is not None and caps.kind == "synthetic":
            return synthetic_detector
        return camera_detector

    fast = FastLoop(pick_detector, settings.tuning)            # no ledger, snapshots only

    judge = judge or build_judge(settings)
    slow = SlowLoop(ledger, judge, settings.tuning)            # writes the ledger: take settlement, scene facts

    takes = TakeStore(settings.data_dir / "takes")
    frames = FrameSource(settings)

    def info() -> dict[str, Any]:
        return {
            "judge": judge.name,
            "judge_is_mock": hasattr(judge, "force_next"),
            "judge_forced": getattr(judge, "forced", None),
            "upload_fps": settings.upload_fps,
            "plan_model": planner.generator_name,    # None = no model configured, example list only
        }

    session = Session(
        read_ledger=ledger.snapshot,                           # read-only
        read_log=ledger.recent,                                # read-only
        planner=planner, fast=fast, slow=slow, takes=takes,
        settings=settings, info=info,
    )
    return Services(
        settings=settings, ledger=ledger, planner=planner, judge=judge,
        camera_detector=camera_detector, fast=fast, slow=slow, takes=takes,
        frames=frames, session=session,
    )
