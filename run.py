"""Entry point.

    python run.py                      # default: YOLO if it loads, else text prompts for detection-based constraints; mock semantic judge
    python run.py --detector none      # explicitly load no detection model (to get "picture in, text prompt out" running first)
    python run.py --judge cosmos       # real Cosmos as the semantic judge (reads the COSMOS3_REASON_URL environment variable)
    python run.py --fresh              # clear progress and start again from the first shot

Open http://127.0.0.1:8000 . The video source can be switched to "Simulated" at the top right: no camera or model needed.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from shotagent.config import Settings
from shotagent.server import create_app
from shotagent.wiring import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Live shooting agent (skeleton)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--detector", choices=["auto", "yolo", "none"], default=None)
    parser.add_argument("--judge", choices=["mock", "cosmos"], default=None)
    parser.add_argument("--yolo-model", default=None, help="detection model, default yolo11n.pt")
    parser.add_argument("--yolo-pose-model", default=None, help='pose model, default yolo11n-pose.pt; pass "" to load none')
    parser.add_argument("--yolo-device", default=None, help='e.g. "mps" (Apple silicon) or "cpu"')
    parser.add_argument("--fps", type=float, default=None, help="frames per second the browser uploads, default 3")
    parser.add_argument("--take-seconds", type=float, default=None,
                        help="fixed length of every take in seconds (overrides the plan's 5)")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true", help="clear the progress in the ledger")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = Settings.from_env(
        detector=args.detector, judge=args.judge, yolo_model=args.yolo_model,
        yolo_pose_model=args.yolo_pose_model, yolo_device=args.yolo_device,
        upload_fps=args.fps, take_seconds=args.take_seconds, data_dir=args.data_dir,
    )
    services = build(settings)
    if args.fresh:
        services.planner.restart()

    log = logging.getLogger("shotagent")
    log.info("Detector: %s %s", services.camera_detector.name, services.camera_detector.note)
    log.info("Semantic judge: %s", services.judge.name)
    log.info("Shot list model: %s", services.planner.generator_name or "none configured (set OPENAI_API_KEY to enable); example list only")
    log.info("Data directory: %s", settings.data_dir)
    log.info("Open http://%s:%d", "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host, args.port)
    uvicorn.run(create_app(services=services), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
