"""Small helpers for encoding and decoding images / video.

Dependencies: OpenCV and the standard library only; imports no internal module.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG bytes -> BGR image. Raises ValueError if it cannot be decoded (empty data too: browsers send empty frames before the camera has a picture)."""
    if not data:
        raise ValueError("Received empty data, not an image")
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError("Not valid image data") from exc
    if image is None:
        raise ValueError("Not valid image data")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def list_frames(clip_dir: str | Path) -> list[Path]:
    return sorted(Path(clip_dir).glob("*.jpg"))


def encode_mp4(clip_dir: str | Path, fps: float, out_name: str = "clip.mp4") -> Path:
    """Join a take directory's 000.jpg, 001.jpg... into an mp4 (the real Cosmos endpoint takes video).

    Prefers ffmpeg (H.264, the most compatible); falls back to OpenCV's own encoder when ffmpeg is not installed.
    """
    clip_dir = Path(clip_dir)
    frames = list_frames(clip_dir)
    if not frames:
        raise ValueError(f"No frames in the take directory: {clip_dir}")
    out = clip_dir / out_name
    fps = max(float(fps), 1.0)

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{fps:.3f}",
            "-i", str(clip_dir / "%03d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # H.264 needs even width and height
            "-movflags", "+faststart",
            str(out),
        ]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
        # If ffmpeg fails (e.g. built without libx264), fall through to the OpenCV path below

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"Cannot read frame: {frames[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for path in frames:
            image = cv2.imread(str(path))
            if image is None:
                continue
            if image.shape[:2] != (h, w):
                image = cv2.resize(image, (w, h))
            writer.write(image)
    finally:
        writer.release()
    if not out.exists() or out.stat().st_size == 0:
        raise ValueError("mp4 encoding failed")
    return out
