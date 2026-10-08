"""Request format and reply parsing of the real Cosmos client, tested against a fake server, never the real endpoint.

The request format follows .cursor/skills/gpu/README.md in the organisers' starter repo. What is guaranteed here is
"what we send matches that document, and replies of every shape get parsed", not how the real model behaves.
"""
from __future__ import annotations

import base64
import json

import httpx
import pytest

from shotagent import media
from shotagent.config import Settings
from shotagent.contracts import JudgeRequest, SemanticQuestion
from shotagent.judge import build_judge
from shotagent.judge.base import JudgeError
from shotagent.judge.cosmos import CosmosJudge
from shotagent.perception.synthetic import SimScene, render

GOOD_REPLY = json.dumps({
    "answers": [{"id": "S02.c4", "pass": True, "reason": "Both are visible in the same room."}],
    "description": "A person sits at a table with a bag underneath.",
    "subjects": {"person": "seated, looking away", "bag": "red bag under the table", "table": None},
})


@pytest.fixture
def request_for_clip(tmp_path) -> JudgeRequest:
    for index in range(6):
        (tmp_path / f"{index:03d}.jpg").write_bytes(media.encode_jpeg(render(SimScene(person=0.7))))
    return JudgeRequest(
        take_id="S02-T1", shot_id="S02", shot_title="The person who has no idea", shot_description="", shot_intent="",
        questions=(SemanticQuestion(
            constraint_id="S02.c4", text="A viewer can tell the person and the bag are in the same space",
            ask="Can a viewer tell that the person and the bag are in the same space?"),),
        subjects={"bag": "bag", "person": "person", "table": "table"},
        clip_dir=str(tmp_path), fps=3.0,
    )


def judge_with(reply_content=GOOD_REPLY, *, model="", status=200, seen=None, models_status=200) -> CosmosJudge:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(models_status, json={"data": [{"id": "nvidia/cosmos3-reason"}]})
        if seen is not None:
            seen.update(path=request.url.path, auth=request.headers.get("authorization"),
                        body=json.loads(request.content))
        return httpx.Response(status, json={"choices": [{"message": {"content": reply_content}}]})

    return CosmosJudge("http://gpu-host:9000/", model=model,
                       client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_request_matches_the_organizers_documented_call(request_for_clip):
    seen: dict = {}
    judge_with(seen=seen).judge(request_for_clip)

    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] is None                                   # the docs: no auth header needed
    body = seen["body"]
    assert body["model"] == "nvidia/cosmos3-reason"               # asked from /v1/models when no model name is configured
    parts = body["messages"][-1]["content"]
    assert [part["type"] for part in parts] == ["text", "video_url"]

    url = parts[1]["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(url.split(",", 1)[1])[4:8] == b"ftyp"   # really an mp4

    prompt = parts[0]["text"]
    assert '"S02.c4"' in prompt and "Can a viewer tell" in prompt   # the question as written in ask, with the constraint id
    assert "person, table" in prompt or "bag, person, table" in prompt


def test_reply_becomes_a_verdict(request_for_clip):
    verdict = judge_with().judge(request_for_clip)
    assert [(a.constraint_id, a.passed) for a in verdict.answers] == [("S02.c4", True)]
    assert verdict.answers[0].reason.startswith("Both are visible")
    assert verdict.description.startswith("A person sits")
    assert verdict.subject_notes == {"person": "seated, looking away", "bag": "red bag under the table"}
    assert verdict.model == "nvidia/cosmos3-reason"


def test_configured_model_is_used_without_asking(request_for_clip):
    seen: dict = {}
    judge_with(model="my/model", seen=seen, models_status=500).judge(request_for_clip)
    assert seen["body"]["model"] == "my/model"


@pytest.mark.parametrize("wrapped", [
    "<think>\nThe bag is under the table...\n</think>\n\n<answer>\n" + GOOD_REPLY + "\n</answer>",
    "Sure, here is the result:\n```json\n" + GOOD_REPLY + "\n```",
    "<think>{not json in here}</think>" + GOOD_REPLY,
])
def test_reply_parsing_tolerates_reasoning_tags_and_chatter(request_for_clip, wrapped):
    verdict = judge_with(wrapped).judge(request_for_clip)
    assert verdict.answers[0].passed is True


@pytest.mark.parametrize("value, expected", [(False, False), ("false", False), ("no", False),
                                              ("true", True), ("yes", True), (None, False)])
def test_pass_field_is_read_strictly(request_for_clip, value, expected):
    reply = json.dumps({"answers": [{"id": "S02.c4", "pass": value}], "description": ""})
    assert judge_with(reply).judge(request_for_clip).answers[0].passed is expected


@pytest.mark.parametrize("bad_reply", [
    "I cannot view videos.",
    json.dumps({"answers": [], "description": "nothing"}),          # the one that had to be answered is missing
    json.dumps({"answers": [{"id": "some other constraint", "pass": True}]}),
    "{ broken json",
])
def test_unusable_reply_is_an_error_not_a_verdict(request_for_clip, bad_reply):
    with pytest.raises(JudgeError):
        judge_with(bad_reply).judge(request_for_clip)


def test_http_failure_is_a_judge_error(request_for_clip):
    with pytest.raises(JudgeError):
        judge_with(status=500).judge(request_for_clip)
    with pytest.raises(JudgeError):
        judge_with(models_status=404).judge(request_for_clip)       # the model name cannot be fetched


def test_missing_endpoint_is_reported_at_startup(tmp_path):
    with pytest.raises(ValueError, match="COSMOS3_REASON_URL"):
        build_judge(Settings(data_dir=tmp_path, judge="cosmos", cosmos_url=""))
