"""Every tunable parameter lives here.

Dependencies: other modules only read config; config imports no internal module.
On-site tuning (thresholds, frame rate, model names, endpoints) means editing this file
or the matching command-line flag / environment variable, nothing else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Tuning:
    """Thresholds for relation checks and take settlement."""

    # -- Relative position (the anchor's box is the ruler, never pixels) --
    under_min_v: float = 0.40        # "under": the subject's centre is at least this far down the anchor box's height
    under_margin_u: float = 0.15     # "under": how far past the anchor box's width it may sit, as a ratio
    above_max_v: float = 0.15        # a subject centre above this ratio counts as "above"
    center_band: float = 0.08        # |u - 0.5| below this counts as "middle": left/right is unclear

    # -- Pose (coarse facing, see spatial.facing_from_keypoints) --
    kp_min_conf: float = 0.30        # keypoints below this confidence are treated as not visible
    kp_pair_conf: float = 0.50       # both points of an ear / eye pair must be at least this confident to set the head's midline
    facing_ears_ratio: float = 0.60  # nose further from the ears' midpoint than this share of half the ear distance = turned that way (about 30°)
    facing_eyes_ratio: float = 0.50  # same with the eyes (about 45°; eyes are closer and jitter more, so the threshold sits between measured frontal and profile values)

    # -- Take settlement --
    pass_ratio: float = 0.60         # a fast constraint passes when at least this share of the take's frames satisfy it
    confirm_below: float = 0.60      # scene facts below this confidence are flagged "please confirm" in the UI


# YOLO (COCO) class name -> role name in this project.
# If your bag is recognised as some other class, add a line here.
# The roles on the right-hand side are the whole cast and prop list the model may use when it writes a shot list (see plan_gen.py).
# The fast loop keeps only the roles the current list uses, so listing more here does not put unrelated boxes on screen.
DEFAULT_LABEL_MAP: dict[str, str] = {
    "person": "person",
    "backpack": "bag",
    "handbag": "bag",
    "suitcase": "bag",
    "dining table": "table",
    # Not used by the hand-written example: common props for generated shot lists
    "chair": "chair",
    "couch": "couch",
    "bed": "bed",
    "cup": "cup",
    "bottle": "bottle",
    "laptop": "laptop",
    "cell phone": "phone",
    "book": "book",
    "umbrella": "umbrella",
    "clock": "clock",
    "potted plant": "plant",
    "vase": "vase",
    "tv": "tv",
    "keyboard": "keyboard",
    "cat": "cat",
    "dog": "dog",
}

# Role name -> default display name (a generated list may rename its roles, e.g. call cup "coffee cup")
ROLE_NAMES: dict[str, str] = {
    "person": "person", "bag": "bag", "table": "table", "chair": "chair", "couch": "couch", "bed": "bed",
    "cup": "cup", "bottle": "bottle", "laptop": "laptop", "phone": "phone", "book": "book",
    "umbrella": "umbrella", "clock": "clock", "plant": "plant", "vase": "vase", "tv": "TV", "keyboard": "keyboard",
    "cat": "cat", "dog": "dog",
}


@dataclass
class Settings:
    # -- Storage --
    data_dir: Path = PROJECT_ROOT / "data"
    web_dir: Path = PROJECT_ROOT / "web"

    # -- Upload from the video source (the browser samples frames with these, see the reply of /api/source) --
    upload_fps: float = 3.0           # design doc: 2 to 5 frames per second
    frame_width: int = 640
    jpeg_quality: float = 0.7
    max_frame_bytes: int = 2_000_000

    # -- Shooting --
    take_seconds: float = 0.0         # above 0, overrides each shot's fixed take length (raise it when one person both shoots and acts)
    min_take_frames: int = 3          # takes with fewer frames are not checked: the user is asked to reshoot

    # -- Fast loop: detector --
    detector: str = "auto"            # auto | yolo | none
    yolo_model: str = "yolo11n.pt"
    yolo_pose_model: str = "yolo11n-pose.pt"   # empty string = no pose model (facing constraints degrade to text prompts)
    yolo_conf: float = 0.25
    yolo_imgsz: int = 640
    yolo_device: str = ""             # empty = let ultralytics choose; "mps" on Apple silicon
    label_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABEL_MAP))

    # -- Slow loop: semantic judge --
    judge: str = "mock"               # mock | cosmos
    mock_latency_s: float = 1.5       # how long the mock pretends to think, so the UI shows "checking"
    # Same names as the environment variables on the organisers' VM (starter repo, config.example): nothing to configure there
    cosmos_url: str = ""              # $COSMOS3_REASON_URL, e.g. http://host:port
    cosmos_model: str = ""            # $COSMOS3_REASON_MODEL; empty = ask /v1/models
    cosmos_timeout_s: float = 90.0
    cosmos_max_tokens: int = 700

    # -- Planner: the LLM that turns an idea / script into a shot list (OpenAI-compatible chat/completions) --
    # Without a key only the hand-written example list is available. To use the organisers' model on site, change these three (all settable by environment variable).
    llm_api_key: str = ""             # $SHOTAGENT_LLM_API_KEY, else $OPENAI_API_KEY
    llm_base_url: str = "https://api.openai.com/v1"   # $SHOTAGENT_LLM_BASE_URL, else $OPENAI_BASE_URL
    llm_model: str = "gpt-5.6-luna"   # $SHOTAGENT_LLM_MODEL
    llm_timeout_s: float = 90.0
    plan_max_shots: int = 6           # most shots a generated list may have
    plan_attempts: int = 2            # how many times to ask (first try included) when the model's list fails validation

    tuning: Tuning = field(default_factory=Tuning)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        env = {
            "cosmos_url": os.environ.get("COSMOS3_REASON_URL", ""),
            "cosmos_model": os.environ.get("COSMOS3_REASON_MODEL", ""),
        }
        for name, variables in (
            ("llm_api_key", ("SHOTAGENT_LLM_API_KEY", "OPENAI_API_KEY")),
            ("llm_base_url", ("SHOTAGENT_LLM_BASE_URL", "OPENAI_BASE_URL")),
            ("llm_model", ("SHOTAGENT_LLM_MODEL",)),
        ):
            value = next((os.environ[v] for v in variables if os.environ.get(v)), "")
            if value:
                env[name] = value
        if os.environ.get("SHOTAGENT_DATA_DIR"):
            env["data_dir"] = Path(os.environ["SHOTAGENT_DATA_DIR"])
        env.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**env)
