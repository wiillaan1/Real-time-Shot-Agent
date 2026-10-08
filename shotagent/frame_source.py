"""Frame intake (server side): turn whatever a video source sends into the Frame the backend understands.

Design doc 2.6: video sources are isolated from the backend. The intake layer hands over three things only:
    one image + a timestamp + the source's capabilities (kind, frame size, sensors)
The fast loop reads just these three and does not care whether the picture is a laptop webcam, the simulated picture, or an RTMP stream later on.

    register(caps)            the source introduces itself; the reply says how it should upload
    ingest(jpeg, ts, sensors) each frame: decode, number, build a Frame

The browser-side counterpart is web/js/sources.js (WebcamSource / SimSource).
To add a source (say the GO Ultra over RTMP): write a process that pulls the stream, samples frames and calls
/api/source and /api/frame like the browser does. Neither this file nor the loops need to change.

Dependencies: contracts, config, media.
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Optional

from . import media
from .config import Settings
from .contracts import Frame, SourceCaps


class NoSource(Exception):
    """A frame arrived before any video source registered (e.g. the server restarted). The client should register again."""


class FrameSource:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._seq = 0
        self.caps: Optional[SourceCaps] = None

    def register(self, caps: SourceCaps) -> dict[str, Any]:
        self.caps = caps
        return {
            "fps": self._settings.upload_fps,
            "width": self._settings.frame_width,
            "jpeg_quality": self._settings.jpeg_quality,
        }

    def ingest(self, jpeg: bytes, ts: Optional[float] = None,
               sensors: Optional[Mapping[str, float]] = None) -> Frame:
        if self.caps is None:
            raise NoSource()
        if len(jpeg) > self._settings.max_frame_bytes:
            raise ValueError(f"This frame is too large ({len(jpeg)} bytes): lower the resolution or the quality")
        image = media.decode_jpeg(jpeg)

        # Readings that come with a frame are ignored unless the source declared the sensor
        readings: dict[str, float] = {}
        if self.caps.sensors and sensors:
            readings = {str(k): float(v) for k, v in sensors.items() if isinstance(v, (int, float))}

        self._seq += 1
        now = time.time()
        return Frame(
            seq=self._seq, ts=float(ts) if ts else now, received_at=now,
            jpeg=jpeg, image=image, sensors=readings,
        )
