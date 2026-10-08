"""HTTP server: a thin shell around Session with no business logic of its own.

    GET  /                       the UI
    POST /api/source             a video source introduces itself (capabilities) -> upload parameters + full state
    POST /api/frame              send one frame (body is a JPEG; X-Frame-Ts capture time, X-Sensors sensor readings)
                                 -> the framing hint for this frame + brief status
    GET  /api/state              full state: session + ledger
    POST /api/shot/start         roll
    POST /api/shot/stop          end the take by hand
    POST /api/shot/select        pick a shot by hand (reshoot one that passed)
    POST /api/reset              clear progress and start over (the list stays)
    POST /api/plan               switch shot lists: {"request": idea or script}; empty = back to the example. Waits on the model for ten seconds or more
    GET  /api/takes/{id}/thumb   thumbnail of a take in the ledger index

    For development:
    GET  /api/sim/frame          render one simulated picture (used by the UI's "Simulated" source)
    POST /api/debug/mock-judge   make the mock pass or fail the next take

Why one HTTP request per frame rather than a WebSocket: the browser waits for the previous reply before sending the next frame,
which is "process only the latest frame" for free, with no dropping or queueing on the server; at 2 to 5 frames per second the
overhead is negligible; and every endpoint can be tried by itself with curl.

All routes are async def: Session methods must be called on the event-loop thread (see the concurrency note in session.py).

Dependencies: wiring, session, frame_source, contracts, config, takes, perception.synthetic, media, plan_gen (for one length limit).
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import media
from .config import Settings
from .contracts import SourceCaps
from .frame_source import NoSource
from .plan_gen import MAX_REQUEST_CHARS
from .perception.synthetic import SimScene, render
from .session import SessionError
from .takes import thumbnail
from .wiring import Services, build


class SelectBody(BaseModel):
    shot_id: Optional[str] = None


class PlanBody(BaseModel):
    request: str = Field(default="", max_length=MAX_REQUEST_CHARS)   # empty = the example list


class MockJudgeBody(BaseModel):
    next: Optional[Literal["pass", "fail"]] = None    # None = cancel


def create_app(settings: Optional[Settings] = None, services: Optional[Services] = None) -> FastAPI:
    services = services or build(settings or Settings.from_env())
    settings = services.settings
    session = services.session

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session.boot()
        yield

    app = FastAPI(title="Live Shooting Agent", lifespan=lifespan)
    app.state.services = services

    @app.exception_handler(SessionError)
    async def session_error(request: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        # During development, edited front-end files should take effect on refresh
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # ───────────── Video source -> frame ─────────────

    @app.post("/api/source")
    async def register_source(caps: SourceCaps) -> dict[str, Any]:
        upload = services.frames.register(caps)
        session.set_source(caps)
        return {"upload": upload, "state": session.view()}

    @app.post("/api/frame")
    async def push_frame(request: Request) -> Any:
        body = await request.body()
        try:
            ts = float(request.headers.get("x-frame-ts", "") or 0) or None
            sensors = json.loads(request.headers.get("x-sensors", "") or "{}")
            if not isinstance(sensors, dict):
                raise ValueError("X-Sensors should be a JSON object")
            frame = services.frames.ingest(body, ts, sensors)
        except NoSource:
            return JSONResponse({"error": "no_source"}, status_code=409)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await session.on_frame(frame)

    # ───────────── State and actions ─────────────

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return session.view()

    @app.post("/api/shot/start")
    async def start() -> dict[str, Any]:
        session.start_take()
        return session.view()

    @app.post("/api/shot/stop")
    async def stop() -> dict[str, Any]:
        session.stop_take("manual")
        return session.view()

    @app.post("/api/shot/select")
    async def select(body: SelectBody) -> dict[str, Any]:
        session.select_shot(body.shot_id)
        return session.view()

    @app.post("/api/reset")
    async def reset() -> dict[str, Any]:
        session.reset()
        return session.view()

    @app.post("/api/plan")
    async def plan(body: PlanBody) -> dict[str, Any]:
        await session.replan(body.request)
        return session.view()

    @app.get("/api/takes/{take_id}/thumb")
    async def take_thumb(take_id: str) -> FileResponse:
        # Only takes in the ledger index can be viewed: take_id is looked up in the ledger, never joined into a path
        take = services.ledger.snapshot().find_take(take_id)
        path = thumbnail(take.clip_dir) if take else None
        if path is None:
            raise HTTPException(status_code=404, detail="The ledger index has no such take")
        return FileResponse(path, media_type="image/jpeg")

    # ───────────── For development ─────────────

    @app.get("/api/sim/frame")
    async def sim_frame(
        table: bool = True,
        bag: Optional[float] = None,
        bag_under: bool = True,
        person: Optional[float] = None,
        facing: str = "front",
        pitch: float = 0.0,
    ) -> Response:
        scene = SimScene(table=table, bag=bag, bag_under=bag_under, person=person, facing=facing, pitch=pitch)
        return Response(media.encode_jpeg(render(scene), quality=90), media_type="image/jpeg")

    @app.post("/api/debug/mock-judge")
    async def mock_judge(body: MockJudgeBody) -> dict[str, Any]:
        judge = services.judge
        if not hasattr(judge, "force_next"):
            raise HTTPException(status_code=404, detail="The current judge is not the mock")
        judge.force_next(None if body.next is None else body.next == "pass")
        return session.view()

    # ───────────── UI ─────────────

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(settings.web_dir / "index.html")

    app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")
    return app
