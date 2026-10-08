"""Fake GPU endpoints: get the --judge cosmos path running before you have the organisers' real ones.

    python scripts/fake_endpoints.py                 # default port 9001
    python scripts/fake_endpoints.py --delay 3       # pretend the model thinks for 3 seconds
    python scripts/fake_endpoints.py --verdict fail  # answer every yes/no question with fail
    python scripts/fake_endpoints.py --think         # wrap replies in <think>...</think><answer>...</answer>

Then in another terminal:

    COSMOS3_REASON_URL=http://127.0.0.1:9001 python run.py --judge cosmos

It can also stand in for the LLM that writes shot lists (same /v1/chat/completions; it tells who is asking from the prompt):

    OPENAI_BASE_URL=http://127.0.0.1:9001/v1 OPENAI_API_KEY=fake python run.py

Whatever you then type under "Change the script" in the UI, the same hard-coded list comes back (it uses only person, bag
and table, so it can be shot to the end in the simulated picture). It is for walking "idea -> list -> shoot" with no key and no network.

The endpoint shapes follow .cursor/skills/gpu/README.md in the organisers' starter repo:
    Cosmos3-Reason   GET /v1/models, /v1/health/ready; POST /v1/chat/completions
    YOLO11           GET /healthz; POST /v1/infer   {"video_base64", "filename", "include_frames"}
    Cosmos Embed1    POST /v1/embeddings            -> 256 dimensions
(In the real environment each model has its own address; here they share one port for convenience.)

Everything returned is hard-coded: it never looks at the picture. The chat/completions reply uses the common format of
OpenAI-compatible APIs; the /v1/infer reply only has the field names the docs mention (perception_ok / object_classes /
object_counts / frames) and their inner structure is a guess. Look at the real one with scripts/probe_endpoints.py.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re

import uvicorn
from fastapi import FastAPI, Request

MODEL = "nvidia/cosmos3-reason"

# The list returned when standing in for the planner's LLM. Format: the prompt in shotagent/plan_gen.py.
CANNED_PLAN = {
    "title": "(fake endpoint) The bag left under the table",
    "synopsis": "Someone leaves a bag under the table. They come back for it, never look under the table, and leave empty-handed. This list is hard-coded by the fake endpoint and has nothing to do with what you typed.",
    "subjects": {"person": "owner", "bag": "backpack", "table": "long table"},
    "shots": [
        {"title": "The forgotten bag", "description": "Empty shot: the long table and the backpack under it, nobody there.", "intent": "Show the audience where the bag is first.",
         "technique": "Establishing shot", "setup": "Setup A (long table, front, medium shot)",
         "checks": [{"predicate": "in_frame", "args": {"a": "bag"}},
                    {"predicate": "under", "args": {"a": "bag", "b": "table"}}],
         "semantic": [{"text": "The bag under the table is visible at a glance", "ask": "Is a bag clearly visible under the table?"}],
         "notes": [], "remember": [{"subject": "bag", "anchor": "table"}]},
        {"title": "Looking in the wrong place", "description": "The owner walks in and looks around at the far end of the long table.", "intent": "They are far from the bag and never look its way.",
         "technique": "Suspense: the audience knows, the character does not", "setup": "Setup A (long table, front, medium shot)",
         "checks": [{"predicate": "in_frame", "args": {"a": "person"}},
                    {"predicate": "apart", "args": {"a": "person", "b": "bag", "ruler": "table", "min_ratio": 0.4}},
                    {"predicate": "not_facing", "args": {"a": "person", "b": "bag"}}],
         "semantic": [{"text": "They are clearly looking for something", "ask": "Does the person appear to be searching for something?"}],
         "notes": ["Move slowly, as if trying to remember where the bag was left"], "remember": []},
        {"title": "Leaving empty-handed", "description": "From a high angle: the owner pauses, turns and walks out of frame.", "intent": "Seen from above, they look helpless.",
         "technique": "High-angle shot", "setup": "Setup B (high angle)",
         "checks": [{"predicate": "in_frame", "args": {"a": "person"}},
                    {"predicate": "pitch_at_least", "args": {"min_deg": 20}}],
         "semantic": [], "notes": ["Hold for a beat before leaving the frame"], "remember": []},
    ],
}


def create_app(delay: float = 0.0, verdict: str = "pass", think: bool = False) -> FastAPI:
    app = FastAPI(title="fake GPU endpoints")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}

    @app.get("/v1/health/ready")
    async def ready():
        return {"status": "ready"}

    @app.get("/v1/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "model_loaded": True, "cuda_available": False, "fake": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        text, video_bytes = "", 0
        for message in body.get("messages", []):
            content = message.get("content")
            for part in content if isinstance(content, list) else []:
                if part.get("type") == "text":
                    text += part.get("text", "")
                elif part.get("type") == "video_url":
                    url = part["video_url"]["url"]
                    video_bytes = len(base64.b64decode(url.split(",", 1)[1])) if url.startswith("data:") else -1
            if isinstance(content, str):
                text += content

        await asyncio.sleep(delay)
        if "planning layer of a live shooting assistant" in text:      # the planner is asking: return the hard-coded list
            return {
                "id": "chatcmpl-fake", "object": "chat.completion", "model": body.get("model", "fake-planner"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": json.dumps(CANNED_PLAN, ensure_ascii=False)}}],
            }
        ids = re.findall(r'id "([^"]+)"', text)
        subjects = re.search(r"not visible: ([^.]+)\.", text)
        answer = json.dumps({
            "answers": [{"id": i, "pass": verdict == "pass",
                         "reason": f"(fake endpoint) fixed answer: {verdict}"} for i in ids],
            "description": f"(fake endpoint) received a {video_bytes}-byte video; nothing was actually watched.",
            "subjects": {name.strip(): "(fake endpoint) placeholder"
                         for name in (subjects.group(1).split(",") if subjects else []) if name.strip() != "(none)"},
        })
        if not ids and "JSON" not in text:        # a plain smoke test: "Reply with the single word: OK"
            answer = "OK"
        if think:
            answer = f"<think>\n(fake reasoning)\n</think>\n\n<answer>\n{answer}\n</answer>"
        return {
            "id": "chatcmpl-fake", "object": "chat.completion", "model": body.get("model", MODEL),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        }

    @app.post("/v1/infer")
    async def infer(request: Request):
        body = await request.json()
        size = len(base64.b64decode(body.get("video_base64", "")))
        await asyncio.sleep(delay)
        return {
            "perception_ok": True, "fake": True, "filename": body.get("filename"), "video_bytes": size,
            "object_classes": ["person", "dining table", "backpack"],
            "object_counts": {"person": 1, "dining table": 1, "backpack": 1},
            "frames": [] if not body.get("include_frames") else [
                {"note": "structure is a guess: see what a real frame entry holds with probe_endpoints.py"}],
        }

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        await request.json()
        return {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 256}]}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake Cosmos / YOLO / Embed endpoints")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds each inference pretends to take")
    parser.add_argument("--verdict", choices=["pass", "fail"], default="pass")
    parser.add_argument("--think", action="store_true", help="wrap replies in <think>/<answer> tags")
    args = parser.parse_args()
    print(f"Fake endpoints at http://127.0.0.1:{args.port} : use it as the address of all three models")
    uvicorn.run(create_app(args.delay, args.verdict, args.think), host="127.0.0.1", port=args.port, log_level="warning")
