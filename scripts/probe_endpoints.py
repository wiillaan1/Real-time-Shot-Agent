"""Run this on first contact with the organisers' GPU endpoints: it answers the design doc's open questions about Cosmos / YOLO in one go.

    python scripts/probe_endpoints.py                               # uses a simulated clip
    python scripts/probe_endpoints.py --clip data/takes/S02/S02-T1_...   # uses a take you really shot

It reads the same environment variables as the organisers' VM (config.example in the starter repo):
    COSMOS3_REASON_URL  COSMOS3_REASON_MODEL  YOLO_URL  COSMOS_EMBED1_URL  COSMOS_EMBED1_MODEL
On the VM they are already set; running it on your own laptop checks whether the endpoints can be reached from the laptop directly.

It tells you:
  1. Cosmos: whether it is reachable; it sends one request in the real format with this project's CosmosJudge and shows what the model
     replied, whether that parses into verdicts, and how long the round trip took (= how long the user waits in place after a take)
  2. YOLO: whether it is reachable; it saves /openapi.json and the full reply of one /v1/infer call. Read those two files for the
     real structure before writing a remote detector for the fast loop (a new Detector under perception/)
  3. Embed1: whether a text embedding has 256 dimensions (for semantic search later)

It prints variable names only, never endpoint addresses; raw model replies go to the --out directory (default probe_out/, already in .gitignore).
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from shotagent import media  # noqa: E402
from shotagent.contracts import JudgeRequest, SemanticQuestion  # noqa: E402
from shotagent.judge.base import JudgeError  # noqa: E402
from shotagent.judge.cosmos import CosmosJudge  # noqa: E402
from shotagent.perception.synthetic import SimScene, render  # noqa: E402


def step(title: str) -> None:
    print(f"\n== {title} ==")


def get(client: httpx.Client, url: str) -> str:
    try:
        reply = client.get(url)
        return str(reply.status_code)
    except httpx.HTTPError as exc:
        return f"unreachable ({type(exc).__name__})"


def demo_clip(directory: Path) -> Path:
    """With no real take, make a 5-second one from the simulated picture (a person walking from beside the bag to the far end of the table)."""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(15):
        scene = SimScene(bag=0.2, person=0.35 + index * 0.025, facing="right")
        (directory / f"{index:03d}.jpg").write_bytes(media.encode_jpeg(render(scene), 90))
    return directory


def probe_cosmos(client: httpx.Client, clip: Path, fps: float, out: Path) -> None:
    step("Cosmos3-Reason")
    base = os.environ.get("COSMOS3_REASON_URL", "").rstrip("/")
    if not base:
        print("COSMOS3_REASON_URL is not set, skipping")
        return
    print("GET /v1/models →", get(client, f"{base}/v1/models"),
          "  GET /v1/health/ready ->", get(client, f"{base}/v1/health/ready"))

    judge = CosmosJudge(base, model=os.environ.get("COSMOS3_REASON_MODEL", ""),
                        api_key=os.environ.get("COSMOS_API_KEY", ""), client=client)
    request = JudgeRequest(
        take_id="probe", shot_id="S02", shot_title="probe", shot_description="", shot_intent="",
        questions=(SemanticQuestion(
            constraint_id="S02.c4", text="A viewer can tell the person and the bag are in the same space",
            ask="Can a viewer tell that the person and the bag are in the same space?"),),
        subjects={"person": "person", "bag": "bag", "table": "table"}, clip_dir=str(clip), fps=fps,
    )
    started = time.perf_counter()
    try:
        verdict = judge.judge(request)
    except JudgeError as exc:
        print(f"Failed ({time.perf_counter() - started:.1f} s): {exc}")
        if judge.last_reply:
            (out / "cosmos_reply.txt").write_text(judge.last_reply, encoding="utf-8")
            print(f"The model's raw reply is in {out / 'cosmos_reply.txt'}: if it cannot be parsed, change the prompt or parse_reply in judge/cosmos.py")
        return
    (out / "cosmos_reply.txt").write_text(judge.last_reply, encoding="utf-8")
    print(f"OK, round trip {verdict.latency_ms / 1000:.1f} s (including joining frames into a video), model {verdict.model}")
    for answer in verdict.answers:
        print(f"  {answer.constraint_id}: {'pass' if answer.passed else 'fail'}  {answer.reason}")
    print(f"  description: {verdict.description}")
    print(f"  per role: {verdict.subject_notes}")
    print(f"  raw reply saved to {out / 'cosmos_reply.txt'}")


def probe_yolo(client: httpx.Client, video: Path, out: Path) -> None:
    step("YOLO11")
    base = os.environ.get("YOLO_URL", "").rstrip("/")
    if not base:
        print("YOLO_URL is not set, skipping")
        return
    print("GET /healthz →", get(client, f"{base}/healthz"))
    try:
        schema = client.get(f"{base}/openapi.json")
        if schema.status_code == 200:
            (out / "yolo_openapi.json").write_text(schema.text, encoding="utf-8")
            paths = list(schema.json().get("paths", {}))
            print(f"API schema saved to {out / 'yolo_openapi.json'}, with these paths: {paths}")
            print("  (look for an endpoint that takes a single image: the fast loop could use it per frame)")
        else:
            print("GET /openapi.json →", schema.status_code)
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Could not get the API schema: {exc}")

    payload = {"video_base64": base64.b64encode(video.read_bytes()).decode("ascii"),
               "filename": video.name, "include_frames": True}
    started = time.perf_counter()
    try:
        reply = client.post(f"{base}/v1/infer", json=payload)
        elapsed = time.perf_counter() - started
        (out / "yolo_infer.json").write_text(reply.text, encoding="utf-8")
        print(f"POST /v1/infer -> {reply.status_code}, {elapsed:.1f} s, full reply saved to {out / 'yolo_infer.json'}")
        data = reply.json()
        if isinstance(data, dict):
            print(f"  top-level fields: {list(data)}")
            print(f"  object_classes: {data.get('object_classes')}")
            frames = data.get("frames")
            if isinstance(frames, list) and frames:
                first = frames[0]
                print(f"  frames has {len(frames)} entries, the first: {list(first) if isinstance(first, dict) else type(first).__name__}")
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Failed: {exc}")


def probe_embed(client: httpx.Client) -> None:
    step("Cosmos Embed1")
    base = os.environ.get("COSMOS_EMBED1_URL", "").rstrip("/")
    if not base:
        print("COSMOS_EMBED1_URL is not set, skipping")
        return
    payload = {"input": "a bag under a table", "model": os.environ.get("COSMOS_EMBED1_MODEL", "nvidia/cosmos-embed1"),
               "request_type": "query", "encoding_format": "float"}
    try:
        reply = client.post(f"{base}/v1/embeddings", json=payload)
        reply.raise_for_status()
        print(f"Text embedding dimensions: {len(reply.json()['data'][0]['embedding'])} (the docs say 256)")
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        print(f"Failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the organisers' Cosmos / YOLO / Embed endpoints")
    parser.add_argument("--clip", type=Path, help="a take directory (holding 000.jpg, 001.jpg...); default: a simulated clip")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("probe_out"))
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    names = ["COSMOS3_REASON_URL", "COSMOS3_REASON_MODEL", "YOLO_URL", "COSMOS_EMBED1_URL"]
    print("Environment: " + ", ".join(f"{name} {'set' if os.environ.get(name) else 'missing'}" for name in names))

    clip = args.clip or demo_clip(args.out / "clip")
    if not args.clip:
        print("No --clip given: using a simulated clip (a cartoon schematic). It can only verify connectivity, format and timing; "
              "whether the semantic judgement is any good needs a really shot take.")
    video = media.encode_mp4(clip, args.fps)
    print(f"Clip: {clip}, video {video.stat().st_size / 1024:.0f} KB")

    with httpx.Client(timeout=args.timeout) as client:
        probe_cosmos(client, clip, args.fps, args.out)
        probe_yolo(client, video, args.out)
        probe_embed(client)


if __name__ == "__main__":
    main()
