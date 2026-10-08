"""Client for a text model: OpenAI-compatible chat/completions. The planner uses it to turn an idea / script into a shot list.

    POST {base_url}/chat/completions          Authorization: Bearer {api_key}
    {"model": ..., "messages": [...], "response_format": {"type": "json_object"}}
    the reply is in choices[0].message.content

Only these three fields are sent and no other parameter (temperature, max_tokens...): model generations differ in what they
accept, and the less is sent the less can be refused. JSON mode is not supported everywhere either: after one 400 it is dropped
for good, since plan_gen finds the JSON in the reply by itself anyway.

Defaults to OpenAI (gpt-5.6-luna). To use the organisers' model on site, change three environment variables:
    SHOTAGENT_LLM_BASE_URL   SHOTAGENT_LLM_API_KEY   SHOTAGENT_LLM_MODEL
(The first two fall back to OPENAI_BASE_URL / OPENAI_API_KEY.)

[UNVERIFIED] The request format was tested against a fake server (tests/test_plan_gen.py), never against the real OpenAI.
Run scripts/try_plan.py the first time you use it.

Dependencies: imports no internal module.
"""
from __future__ import annotations

from typing import Optional, Sequence

import httpx


class LLMError(Exception):
    """The model call failed (no key, no network, refused, empty reply)."""


class ChatLLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 90.0,
                 client: Optional[httpx.Client] = None):
        if not api_key:
            raise ValueError("No API key for the model: set the OPENAI_API_KEY (or SHOTAGENT_LLM_API_KEY) environment variable")
        if not model:
            raise ValueError("No model name: set the SHOTAGENT_LLM_MODEL environment variable")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=timeout_s)
        self._json_mode = True
        self.model = model

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        """Send one round of conversation and return the text of the model's reply."""
        reply = self._post(messages)
        if reply.status_code == 400 and self._json_mode:
            # The most common 400 is an endpoint that does not accept response_format: drop it, retry once, and leave it off from then on
            self._json_mode = False
            reply = self._post(messages)
        if reply.status_code >= 400:
            raise LLMError(f"{self.model} refused the request (HTTP {reply.status_code}): {_error_text(reply)}")
        try:
            content = reply.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"{self.model}'s reply is not in the expected format: {reply.text[:200]!r}") from exc
        if not content or not str(content).strip():
            raise LLMError(f"{self.model} returned an empty reply")
        return str(content)

    def _post(self, messages: Sequence[dict[str, str]]) -> httpx.Response:
        payload: dict = {"model": self.model, "messages": list(messages)}
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            return self._client.post(self._url, json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach the model ({type(exc).__name__}): {exc}") from exc


def _error_text(reply: httpx.Response) -> str:
    """The server's reason for the error (OpenAI's format is {"error": {"message": ...}}); the start of the raw body if there is none."""
    try:
        error = reply.json().get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        if isinstance(error, str):
            return error[:300]
    except (ValueError, AttributeError):
        pass
    return reply.text[:300]
