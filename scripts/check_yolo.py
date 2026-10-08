"""Answers the design doc's first open question: can YOLO reliably recognise your bag and table, and how good is the facing estimate.

    python scripts/check_yolo.py                      # default camera, watch for 10 seconds
    python scripts/check_yolo.py --seconds 20
    python scripts/check_yolo.py --camera 1           # the second camera
    python scripts/check_yolo.py --source clip.mp4    # a video
    python scripts/check_yolo.py --image desk.jpg     # a photo
    python scripts/check_yolo.py --model yolo11s.pt   # try again with a larger model
    python scripts/check_yolo.py --save out.jpg       # save the last frame with its detection boxes

Set up the bag, the table and the person as they will be in the demo, then run it. It uses the same detector as the server
(perception/yolo.py), so what is recognised here is recognised in the fast loop. It ends with a stability summary and advice.

It also lists the other classes the model saw: if your bag keeps being recognised as something else,
add a line mapping that class to bag in DEFAULT_LABEL_MAP in shotagent/config.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from shotagent.config import Settings  # noqa: E402
from shotagent.spatial import facing_from_keypoints, relative_position  # noqa: E402

ROLES = {"person": "person", "bag": "bag", "table": "table"}
FACING = {"left": "facing frame left", "right": "facing frame right", "front": "facing the camera", "back": "back to the camera", None: "can't tell"}
SOLID_CONF = 0.5    # average confidence below this: recognised in every frame, but so close to the detection threshold that a change of light loses it


def frames_from(args) -> "tuple[object, bool]":
    """Returns (frame generator, whether it is live)."""
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            sys.exit(f"Cannot read this image: {args.image}")
        return iter([image]), False

    target = args.source if args.source else args.camera
    capture = cv2.VideoCapture(target)
    if not capture.isOpened():
        sys.exit(f"Cannot open the video source: {target} (camera in use by another program? try another --camera number)")

    def generate():
        deadline = time.time() + args.seconds
        try:
            while args.source or time.time() < deadline:
                ok, image = capture.read()
                if not ok:
                    break
                yield image
        finally:
            capture.release()

    return generate(), not args.source


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether YOLO recognises your bag, table and person")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", help="a video file")
    parser.add_argument("--image", help="a single image")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to watch in camera mode")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--device", default="", help='e.g. "mps" or "cpu"')
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", help="save the last frame with its detection boxes as an image")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO

        from shotagent.perception.yolo import YoloDetector
    except ImportError:
        sys.exit("ultralytics is not installed: pip install ultralytics")

    label_map = Settings().label_map
    print(f"Loading {args.model} + {args.pose_model} ...")
    detector = YoloDetector(model=args.model, pose_model=args.pose_model, label_map=label_map,
                            conf=args.conf, device=args.device)
    everything = YOLO(args.model)     # no class filter: see what everything in the picture is recognised as

    frames, live = frames_from(args)
    seen = Counter()                  # in how many frames each role appears
    confidences = defaultdict(list)
    raw_names = defaultdict(Counter)  # role -> the classes it was actually recognised as
    others = Counter()                # classes not mapped to any role
    facings = Counter()
    placements = Counter()
    timings = []
    total = 0
    last = None

    for image in frames:
        total += 1
        started = time.perf_counter()
        found = detector.detect(image)
        timings.append((time.perf_counter() - started) * 1000)

        best = {}
        for detection in found:
            if detection.label not in best or detection.conf > best[detection.label].conf:
                best[detection.label] = detection

        parts = []
        for role, name in ROLES.items():
            detection = best.get(role)
            if detection is None:
                parts.append(f"{name} -")
                continue
            seen[role] += 1
            confidences[role].append(detection.conf)
            raw_names[role][detection.raw_label] += 1
            text = f"{name} {detection.conf:.2f}"
            if role == "bag":
                text += f" ({detection.raw_label})"
            if role == "person":
                facing = facing_from_keypoints(detection.keypoints)
                facings[facing] += 1
                text += f" ({FACING[facing]})"
            parts.append(text)

        if "bag" in best and "table" in best:
            rel = relative_position(best["bag"].box, best["table"].box)
            if rel is not None:
                placements[(rel.vertical, rel.side)] += 1
                parts.append(f"bag vs table u={rel.u:.2f} v={rel.v:.2f} -> {rel.vertical}/{rel.side}")

        result = everything.predict(image, conf=args.conf, device=args.device or None, verbose=False)[0]
        extra = Counter()
        for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            raw = result.names[int(cls)]
            if raw not in label_map:
                extra[raw] = max(extra[raw], round(conf, 2))
        for raw in extra:
            others[raw] += 1
        tail = ", ".join(f"{raw} {conf}" for raw, conf in extra.most_common(4))

        print(f"#{total:<3d} " + "  ".join(parts) + (f"  | other: {tail}" if tail else "") + f"  | {timings[-1]:.0f} ms")
        last = (image, list(best.values()))

    if total == 0:
        sys.exit("Not a single frame was read")

    print("\n--- Summary ---")
    print(f"{total} frames, detection + pose averaged {mean(timings[1:] or timings):.0f} ms per frame"
          + (" (live camera)" if live else ""))
    if total == 1:
        print("  (only one frame: it shows whether things are recognised; stability needs a camera or a video)")
    verdicts = {}
    for role, name in ROLES.items():
        share = seen[role] / total
        average = mean(confidences[role]) if seen[role] else 0.0
        if share >= 0.8 and average >= SOLID_CONF:
            verdicts[role] = "stable"
        elif share >= 0.8:
            verdicts[role] = "recognised, but with low confidence"
        elif share >= 0.4:
            verdicts[role] = "unstable"
        else:
            verdicts[role] = "mostly not recognised"
        detail = ""
        if seen[role]:
            detail = f", average confidence {average:.2f}, recognised as " + ", ".join(
                f"{raw}×{count}" for raw, count in raw_names[role].most_common())
        print(f"  {name}: {seen[role]}/{total} frames ({share:.0%}) -> {verdicts[role]}{detail}")
    if facings:
        print("  person's facing: " + ", ".join(f"{FACING[f]} {count} frames" for f, count in facings.most_common()))
    if placements:
        print("  bag vs table: " + ", ".join(f"{v}/{s} {count} frames" for (v, s), count in placements.most_common()))
    if others:
        print("  also recognised in the picture: " + ", ".join(f"{raw} ({count} frames)" for raw, count in others.most_common(8)))

    print("\n--- Advice ---")
    if all(v == "stable" for v in verdicts.values()):
        print("  All three are stable: good to go.")
    if verdicts["bag"] != "stable":
        print("  The bag is not held reliably: use a bag with a strong colour and a clear outline that stands out from the background; "
              "check whether one of the classes under 'also recognised' is in fact your bag and add it to DEFAULT_LABEL_MAP; "
              "or try a larger model (--model yolo11s.pt).")
    if verdicts["table"] != "stable":
        print("  The table is not held reliably: COCO only has the dining table class, so get both the top and the legs in frame and keep the top uncluttered; "
              "the table is the yardstick for distance, and without it neither distance nor 'under' can be judged.")
    if verdicts["person"] != "stable":
        print("  The person is not held reliably: get the whole upper body in frame and avoid dim light.")

    if args.save and last is not None:
        image, detections = last
        height, width = image.shape[:2]
        for detection in detections:
            box = detection.box
            p1 = (int(box.x1 * width), int(box.y1 * height))
            p2 = (int(box.x2 * width), int(box.y2 * height))
            cv2.rectangle(image, p1, p2, (255, 255, 255), 2)
            cv2.putText(image, f"{detection.label} {detection.conf:.2f}", (p1[0] + 4, max(p1[1] + 18, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(args.save, image)
        print(f"\nLast frame saved to {args.save}")


if __name__ == "__main__":
    main()
