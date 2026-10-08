"""The data structures passed between modules: the project's one shared vocabulary.

Dependencies: this file imports no internal module; every module may import it.
Anything that crosses a module boundary (a frame, a detection, a constraint, a hint, a take, a semantic verdict)
is defined here, so to see exactly what module A hands to module B, this is the only file to read.

The ledger's own state is not here but in ledger/state.py (the ledger's read model).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    """Immutable data object. To "change" one, make a new one with model_copy(update=...)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# Names of perception capabilities. Video-source sensor names (e.g. "gyro") are compared in the same set.
CAP_DETECT = "detect"    # can produce detection boxes
CAP_POSE = "pose"        # can produce human keypoints


# ───────────────────────── Enums ─────────────────────────

class Source(str, Enum):
    """Source tag: every conclusion in the ledger says who it came from."""

    PLAN = "plan"        # written by the planner (the shot list)
    YOLO = "yolo"        # detection boxes / pose + rules
    COSMOS = "cosmos"    # the slow loop's semantic judgement
    SENSOR = "sensor"    # a sensor reading from the video source (e.g. gyro)


class Checker(str, Enum):
    """Who checks a constraint (the "checker" column of the design doc's constraint table)."""

    YOLO = "yolo"        # object detection + rules
    POSE = "pose"        # pose estimation (coarse)
    SENSOR = "sensor"    # video-source sensor
    COSMOS = "cosmos"    # semantic judgement
    NONE = "none"        # a technique nothing can check: text prompt only, labelled honestly


class Loop(str, Enum):
    FAST = "fast"
    SLOW = "slow"


class ShotStatus(str, Enum):
    TODO = "todo"        # not shot yet
    PASSED = "passed"    # passed
    RETAKE = "retake"    # needs a retake


class CheckState(str, Enum):
    OK = "ok"                # satisfied
    VIOLATED = "violated"    # not satisfied
    UNKNOWN = "unknown"      # cannot tell from this frame (subject out of frame, keypoints unclear...)
    DEGRADED = "degraded"    # the current video source / detector lacks the capability: reduced to a text prompt
    PENDING = "pending"      # slow-loop constraint: judged after the take


# ───────────────────────── Video source → frame ─────────────────────────

class SourceCaps(Frozen):
    """What a video source can do. The intake layer only has to hand over one image + a timestamp + this."""

    kind: str = "webcam"             # webcam | synthetic | rtmp | phone ...
    label: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0                 # the source's native frame rate (the upload rate is set by the server at registration, see /api/source)
    sensors: tuple[str, ...] = ()    # e.g. ("gyro",); empty for a laptop webcam


@dataclass(frozen=True)
class Frame:
    """One frame. The fast loop depends on nothing else."""

    seq: int
    ts: float                        # capture time (client clock, epoch seconds)
    received_at: float               # when the server received it (epoch seconds)
    jpeg: bytes
    image: Any                       # decoded BGR image (numpy.ndarray)
    sensors: Mapping[str, float]     # sensor readings that came with the frame, e.g. {"pitch": 32.0}


# ───────────────────────── Perception results ─────────────────────────

class Box(Frozen):
    """A box normalised to [0,1], origin at the top-left of the picture."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return max(self.x2 - self.x1, 0.0)

    @property
    def h(self) -> float:
        return max(self.y2 - self.y1, 0.0)


class Detection(Frozen):
    label: str                       # role name: person | bag | table ... (see config.DEFAULT_LABEL_MAP)
    raw_label: str = ""              # the model's own class name
    conf: float = 1.0
    box: Box
    # Only for person, and only with a pose model: keypoint name -> (x, y, confidence), also normalised
    keypoints: Optional[dict[str, tuple[float, float, float]]] = None


class Observation(Frozen):
    """What was perceived in one frame: detections + the sensor readings that came with it.
    One of the fast loop's outputs; a take stores one per frame."""

    seq: int
    ts: float
    detections: tuple[Detection, ...] = ()
    sensors: dict[str, float] = {}


# ───────────────────────── Plan: shots and constraints ─────────────────────────

class Constraint(Frozen):
    """One checkable constraint. It stores a relation, not a position."""

    id: str
    text: str                        # description for people, e.g. "Person and bag are more than half a table width apart"
    checker: Checker
    loop: Loop
    predicate: str = ""              # name of the fast loop's relation predicate, see constraints.PREDICATES
    args: dict[str, Any] = {}        # predicate arguments, e.g. {"a": "person", "b": "bag", "ruler": "table"}
    needs: tuple[str, ...] = ()      # sensors required when checker=SENSOR
    fix: str = ""                    # hint template when not satisfied; may use the predicate's placeholders ({away} etc.)
    prompt: str = ""                 # the text prompt it degrades to when it cannot be checked automatically
    ask: str = ""                    # when checker=COSMOS: how to put it to the model (a yes/no question)


class Establish(Frozen):
    """After this shot passes, record where which subject sits relative to which anchor in the scene state."""

    subject: str
    anchor: str


class Shot(Frozen):
    id: str
    order: int
    title: str
    description: str                 # what to film
    intent: str                      # narrative intent
    technique: str = ""              # name of the technique
    setup: str = ""                  # camera setup (shots from the same setup are adjacent; moving is prompted only after a pass)
    duration_s: float = 5.0          # length of the fixed-timer take
    constraints: tuple[Constraint, ...] = ()
    establishes: tuple[Establish, ...] = ()
    source: Source = Source.PLAN


class Plan(Frozen):
    style: str                       # title of the list (a style name for the hand-written example, the piece's title for generated lists)
    subjects: dict[str, str]         # role name -> display name, e.g. {"bag": "backpack"}
    shots: tuple[Shot, ...]
    request: str = ""                # the user's idea or script; empty = the hand-written example list
    synopsis: str = ""               # story synopsis (generated lists only)


# ───────────────────────── Fast loop output ─────────────────────────

class CheckResult(Frozen):
    constraint_id: str
    state: CheckState
    message: str = ""                # hint for the user (empty when satisfied)
    readout: str = ""                # the measurement as a short phrase for people, e.g. "0.31 table widths"
    measured: dict[str, Any] = {}    # the same measurement for programs, e.g. {"ratio": 0.31, "min_ratio": 0.5}
    basis: dict[str, str] = {}       # where each operand came from: {"bag": "ledger", "person": "live"}


class Recalled(Frozen):
    """Something not visible in this frame and recovered from the ledger (the UI draws it with a dashed box)."""

    label: str
    box: Box
    take_id: str                     # the take this position fact in the ledger came from


class Guidance(Frozen):
    """The fast loop's output for one frame."""

    seq: int
    shot_id: Optional[str]
    headline: str                    # the one hint to deal with first
    level: str                       # ok | fix | info (the UI colours by it)
    ready: bool                      # whether every automatically checkable fast constraint is satisfied (degraded ones neither block nor count)
    checks: tuple[CheckResult, ...] = ()   # per-constraint results; a degraded constraint's text prompt is in its message
    notes: tuple[str, ...] = ()      # secondary hints (e.g. the picture disagrees with the ledger)
    recalled: tuple[Recalled, ...] = ()   # subjects recovered from the ledger
    observation: Observation
    latency_ms: float = 0.0


# ───────────────────────── Takes ─────────────────────────

class RecordedFrame(Frozen):
    seq: int
    ts: float
    file: str                        # file name inside the take directory
    observation: Observation
    checks: tuple[CheckResult, ...]


class Take(Frozen):
    """A finished take, saved to disk, handed to the slow loop for settlement."""

    take_id: str
    shot_id: str
    started_at: float
    ended_at: float
    end_reason: str                  # timer | manual
    dir: str                         # take directory (absolute path)
    fps: float                       # actual frame rate
    frames: tuple[RecordedFrame, ...]


# ───────────────────────── Semantic judgement (Cosmos) ─────────────────────────

class SemanticQuestion(Frozen):
    constraint_id: str
    text: str                        # the constraint as shown to people
    ask: str = ""                    # how to ask the model; empty = use text


class JudgeRequest(Frozen):
    take_id: str
    shot_id: str
    shot_title: str
    shot_description: str
    shot_intent: str
    questions: tuple[SemanticQuestion, ...]
    subjects: dict[str, str]         # have the model describe these as well
    clip_dir: str                    # directory holding the take's frames
    fps: float


class JudgeAnswer(Frozen):
    constraint_id: str
    passed: bool
    reason: str = ""
    confidence: float = 1.0


class JudgeVerdict(Frozen):
    answers: tuple[JudgeAnswer, ...]
    description: str                 # semantic description of the take (goes into the ledger, for later search)
    subject_notes: dict[str, str] = {}   # role name -> semantic description (the semantic field of the scene state)
    model: str = ""                  # the model that actually judged; the mock honestly writes "mock"
    latency_ms: float = 0.0
