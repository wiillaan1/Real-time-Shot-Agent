"""Takes: collected in memory while recording, saved to disk at the end.

    TakeBuffer   while recording: for each frame the session adds "the frame + the fast loop's result for it"
    TakeStore    after the take: frames are written as 000.jpg, 001.jpg..., with the per-frame detections and check results in take.json

Besides the pictures, a take carries what the fast loop saw in each frame (Observation) and each frame's constraint results.
The slow loop settles from these: fast constraints are not recomputed, and YOLO's positions reach the ledger from here.

Disk layout: data/takes/<shot id>/<take id>_<start time>/
Files of replaced takes stay on disk; they are just no longer in the ledger's index (design doc 2.5, item 2).

In this skeleton a take is the uploaded sampled frames (2 to 5 per second). For full-frame-rate video later,
have the browser record with MediaRecorder and upload into the same directory after the take; settlement does not change.

Dependencies: contracts, media.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from . import media
from .contracts import Frame, Guidance, RecordedFrame, Shot, Take


class TakeBuffer:
    def __init__(self, take_id: str, shot: Shot, started_at: Optional[float] = None):
        self.take_id = take_id
        self.shot = shot
        self.started_at = started_at if started_at is not None else time.time()
        self._started_mono = time.monotonic()
        self._items: list[tuple[Frame, Guidance]] = []

    def add(self, frame: Frame, guidance: Guidance) -> None:
        self._items.append((frame, guidance))

    @property
    def items(self) -> list[tuple[Frame, Guidance]]:
        return list(self._items)

    @property
    def n_frames(self) -> int:
        return len(self._items)

    def elapsed(self) -> float:
        return time.monotonic() - self._started_mono


class TakeStore:
    def __init__(self, root: Path):
        self._root = Path(root)

    def save(self, buffer: TakeBuffer, ended_at: float, end_reason: str) -> Take:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(buffer.started_at))
        directory = self._root / buffer.shot.id / f"{buffer.take_id}_{stamp}"
        suffix = 1
        while directory.exists():   # reopening the same take number within one second (after starting over) must not overwrite files
            suffix += 1
            directory = directory.with_name(f"{buffer.take_id}_{stamp}-{suffix}")
        directory.mkdir(parents=True)

        frames = []
        for index, (frame, guidance) in enumerate(buffer.items):
            name = f"{index:03d}.jpg"
            (directory / name).write_bytes(frame.jpeg)
            frames.append(RecordedFrame(
                seq=frame.seq, ts=frame.ts, file=name,
                observation=guidance.observation, checks=guidance.checks,
            ))

        duration = max(ended_at - buffer.started_at, 1e-3)
        take = Take(
            take_id=buffer.take_id, shot_id=buffer.shot.id,
            started_at=buffer.started_at, ended_at=ended_at, end_reason=end_reason,
            dir=str(directory), fps=round(len(frames) / duration, 2), frames=tuple(frames),
        )
        (directory / "take.json").write_text(take.model_dump_json(indent=1), encoding="utf-8")
        return take


def thumbnail(clip_dir: str | Path) -> Optional[Path]:
    """A take's thumbnail: its middle frame."""
    frames = media.list_frames(clip_dir)
    return frames[len(frames) // 2] if frames else None
