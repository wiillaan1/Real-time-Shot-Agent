"""The real semantic judge: NVIDIA Cosmos3-Reason (OpenAI-compatible chat/completions, video embedded as base64).

The call follows the organisers' starter repo (.cursor/skills/gpu/README.md, version of 2026-09-30):

    POST {COSMOS3_REASON_URL}/v1/chat/completions      no auth header needed
    {"model": "{COSMOS3_REASON_MODEL}",                 if unset, ask GET /v1/models
     "messages": [{"role": "user", "content": [
         {"type": "text", "text": "..."},
         {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}]}],
     "max_tokens": ..., "temperature": 0}
    the reply is in choices[0].message.content

[UNVERIFIED] The request format and the parsing were tested against a fake server (tests/test_judge_cosmos.py)
but never run against the real endpoint. The first time you connect, run scripts/probe_endpoints.py and look at three things:
  - whether the model returns just one JSON object as asked (parsing already tolerates <think>...</think>, <answer>...</answer> and chatter around it)
  - how long a 5-second take takes round trip (that is how long the user waits in place)
  - what the reasons and descriptions that come back look like

Dependencies: contracts, media, judge.base.
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from .. import media
from ..contracts import JudgeAnswer, JudgeRequest, JudgeVerdict
from .base import JudgeError

SYSTEM_PROMPT = (
    "You review one take of a film shot against a checklist. "
    "Judge strictly from what is visible in the clip. Reply with a single JSON object and nothing else."
)


def build_prompt(request: JudgeRequest) -> str:
    lines = ["Checks (answer each one with pass true or false):"]
    for q in request.questions:
        lines.append(f'- id "{q.constraint_id}": {q.ask or q.text}')
    if not request.questions:
        lines.append("- (none)")
    subjects = ", ".join(request.subjects) or "(none)"
    lines += [
        "",
        "Also write one sentence describing the clip (for later search), and a short phrase for each of "
        f"these subjects as it appears in the clip, or null if it is not visible: {subjects}.",
        "",
        "Reply with exactly this JSON shape:",
        '{"answers": [{"id": "<check id>", "pass": true, "reason": "<short reason>"}], '
        '"description": "<one sentence>", "subjects": {"<subject>": "<phrase or null>"}}',
    ]
    return "\n".join(lines)


def parse_reply(content: str, request: JudgeRequest) -> tuple[tuple[JudgeAnswer, ...], str, dict[str, str]]:
    """The model's reply -> (per-question verdicts, take description, per-role descriptions). Raises JudgeError if it cannot be parsed."""
    text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.S)
    tagged = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if tagged:
        text = tagged.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise JudgeError(f"The model's reply contains no JSON: {(content or '')[:200]!r}")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"The JSON in the model's reply cannot be parsed: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeError("The JSON in the model's reply is not an object")

    by_id = {str(a.get("id")): a for a in data.get("answers") or [] if isinstance(a, dict)}
    answers = []
    for q in request.questions:
        answer = by_id.get(q.constraint_id)
        if answer is None:
            raise JudgeError(f"The model's reply has no verdict for {q.constraint_id}")
        answers.append(JudgeAnswer(
            constraint_id=q.constraint_id,
            passed=_truthy(answer.get("pass")),
            reason=str(answer.get("reason") or ""),
        ))

    subjects = data.get("subjects") if isinstance(data.get("subjects"), dict) else {}
    notes = {
        label: str(subjects[label]).strip()
        for label in request.subjects
        if subjects.get(label) not in (None, "", "null")
    }
    return tuple(answers), str(data.get("description") or "").strip(), notes


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "pass", "passed", "1")


class CosmosJudge:
    def __init__(self, base_url: str, model: str = "", api_key: str = "", timeout_s: float = 90.0,
                 max_tokens: int = 700, client: Optional[httpx.Client] = None):
        if not base_url:
            raise ValueError("No Cosmos endpoint configured: set the COSMOS3_REASON_URL environment variable (already set on the organisers' VM)")
        self._base = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout_s, headers=headers)
        self.last_reply = ""    # the model's last raw reply, for tuning the prompt or debugging the parser

    @property
    def name(self) -> str:
        return f"cosmos({self._model or 'model not asked yet'})"

    def _resolve_model(self) -> str:
        if not self._model:
            try:
                reply = self._client.get(f"{self._base}/v1/models")
                reply.raise_for_status()
                self._model = reply.json()["data"][0]["id"]
            except Exception as exc:
                raise JudgeError(f"Could not get the model name (GET /v1/models): {exc}") from exc
        return self._model

    def judge(self, request: JudgeRequest) -> JudgeVerdict:
        started = time.perf_counter()
        try:
            video: Path = media.encode_mp4(request.clip_dir, request.fps)
        except ValueError as exc:
            raise JudgeError(f"The take could not be joined into a video: {exc}") from exc
        encoded = base64.b64encode(video.read_bytes()).decode("ascii")

        payload = {
            "model": self._resolve_model(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": build_prompt(request)},
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}},
                ]},
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0,
        }
        try:
            reply = self._client.post(f"{self._base}/v1/chat/completions", json=payload)
            reply.raise_for_status()
            content = reply.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise JudgeError(f"The Cosmos call failed: {exc}") from exc

        self.last_reply = content or ""
        answers, description, notes = parse_reply(content, request)
        return JudgeVerdict(
            answers=answers, description=description, subject_notes=notes, model=self._model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
