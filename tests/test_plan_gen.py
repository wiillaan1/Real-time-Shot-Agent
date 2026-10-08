"""Idea / script -> shot list: the menu, validation, the retry, and the LLM client's request format.

The model is fake (it returns prepared replies in order). What is guaranteed here is that whatever the model replies,
the list that reaches the ledger can be shot. How well a real model writes is not: run scripts/try_plan.py with a real key for that.
"""
from __future__ import annotations

import copy
import json

import httpx
import pytest

from shotagent.config import DEFAULT_LABEL_MAP, ROLE_NAMES, Settings
from shotagent.constraints import PREDICATES, degraded_prompt
from shotagent.contracts import Checker, Loop
from shotagent.llm import ChatLLM, LLMError
from shotagent.plan_gen import PlanError, PlanGenerator, system_prompt, to_plan
from shotagent.wiring import build_generator

ROLES = {"person": "person", "bag": "bag", "table": "table", "cup": "cup"}
IDEA = "Film someone who forgot their bag under the table"

GOOD = {
    "title": "The bag left under the table",
    "synopsis": "Someone leaves a bag under the table and, coming back for it, never once looks under the table.",
    "subjects": {"person": "owner", "bag": "backpack", "table": "long table", "cup": "unused cup"},
    "shots": [
        {"title": "The forgotten bag", "description": "Empty shot: the long table and the backpack under it.", "intent": "Show the audience where the bag is first.",
         "technique": "Establishing shot", "setup": "Setup A (front, medium shot)",
         "checks": [{"predicate": "in_frame", "args": {"a": "bag"}},
                    {"predicate": "under", "args": {"a": "bag", "b": "table"}}],
         "semantic": [{"text": "The bag under the table is visible at a glance", "ask": "Is a bag clearly visible under the table?"}],
         "remember": [{"subject": "bag", "anchor": "table"}]},
        {"title": "Looking in the wrong place", "description": "The owner looks around at the far end of the long table.", "intent": "They are far from the bag.",
         "setup": "Setup A (front, medium shot)",
         "checks": [{"predicate": "in_frame", "args": {"a": "person"}},
                    {"predicate": "apart", "args": {"a": "person", "b": "bag", "ruler": "table", "min_ratio": 0.4}},
                    {"predicate": "not_facing", "args": {"a": "person", "b": "bag"}}],
         "notes": ["Slower movements"]},
        {"title": "Leaving empty-handed", "description": "High angle: the owner walks out of frame.", "intent": "It looks helpless.",
         "checks": [{"predicate": "pitch_at_least", "args": {"min_deg": 20}}]},
    ],
}


def edited(change) -> dict:
    data = copy.deepcopy(GOOD)
    change(data)
    return data


class Scripted:
    """Fake model: returns prepared replies in order and records every conversation it receives."""

    def __init__(self, *replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in replies]
        self.calls: list[list[dict]] = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.replies[len(self.calls) - 1]


def generator(*replies, **kwargs) -> tuple[PlanGenerator, Scripted]:
    model = Scripted(*replies)
    return PlanGenerator(model, ROLES, **kwargs), model


# ───────────── Menu ─────────────

def test_the_menu_is_built_from_what_the_system_can_actually_check():
    prompt = system_prompt(ROLES, max_shots=6)
    for name in PREDICATES:
        assert f"{name}(" in prompt
    for role, name in ROLES.items():
        assert f"\n  {role}\n" in prompt
    assert "min_ratio=0.5 [0.1..3]" in prompt and "a must be person" in prompt


def test_every_role_the_detector_maps_to_has_a_display_name():
    assert set(DEFAULT_LABEL_MAP.values()) <= set(ROLE_NAMES)


# ───────────── Reply -> Plan ─────────────

def test_a_good_reply_becomes_a_plan_the_loops_can_run():
    make, model = generator(GOOD)
    plan = make("  Film someone who forgot their bag under the table  ")

    assert len(model.calls) == 1 and model.calls[0][1] == {"role": "user", "content": IDEA}
    assert (plan.style, plan.request) == ("The bag left under the table", IDEA) and plan.synopsis
    assert plan.subjects == {"person": "owner", "bag": "backpack", "table": "long table"}     # unused roles stay out of the list
    assert [s.id for s in plan.shots] == ["S01", "S02", "S03"] and [s.order for s in plan.shots] == [1, 2, 3]

    s01, s02, s03 = plan.shots
    assert [c.id for c in s01.constraints] == ["S01.c1", "S01.c2", "S01.c3"]
    assert [(c.checker, c.loop) for c in s01.constraints] == [
        (Checker.YOLO, Loop.FAST), (Checker.YOLO, Loop.FAST), (Checker.COSMOS, Loop.SLOW)]
    assert s01.constraints[1].text == "The backpack is under the long table"    # wording is filled in here from the template, with the story's names
    assert s01.constraints[2].ask == "Is a bag clearly visible under the table?"
    assert [(e.subject, e.anchor) for e in s01.establishes] == [("bag", "table")]

    apart, facing, note = s02.constraints[1:]
    assert apart.args == {"a": "person", "b": "bag", "ruler": "table", "min_ratio": 0.4}
    assert apart.text == "The owner and the backpack are more than 0.4 long table widths apart" and apart.fix == PREDICATES["apart"].fix
    assert facing.checker is Checker.POSE and facing.prompt == PREDICATES["not_facing"].prompt
    assert (note.checker, note.loop, note.prompt) == (Checker.NONE, Loop.FAST, "Slower movements")   # text prompt only

    pitch = s03.constraints[0]
    assert (pitch.checker, pitch.needs, pitch.text) == (Checker.SENSOR, ("gyro",), "Camera pitch ≥ 20°")
    assert s03.setup == "Setup A"                                               # the default when the model gave no setup


def test_numbers_are_clamped_rather_than_sent_back():
    def wild(data):
        data["shots"][1]["checks"][1]["args"]["min_ratio"] = 99
        data["shots"][2]["checks"][0]["args"] = {"min_deg": "very high"}
    plan, errors = to_plan(edited(wild), "x", ROLES, 6)
    assert not errors
    assert plan.shots[1].constraints[1].args["min_ratio"] == 3.0
    assert plan.shots[2].constraints[0].args["min_deg"] == 25                  # an unreadable number falls back to the default


@pytest.mark.parametrize("change, expected", [
    (lambda d: d["shots"][0]["checks"][0]["args"].update(a="gun"), "'gun' is not an available subject id"),
    (lambda d: d["shots"][0]["checks"][0].update(predicate="looks_sad"), "unknown predicate 'looks_sad'"),
    (lambda d: d["shots"][1]["checks"][2]["args"].update(a="bag", b="table"), "not_facing.a must be person"),
    (lambda d: d["shots"][1]["checks"][1]["args"].update(ruler="bag"), "must be different subjects"),
    (lambda d: d["shots"][0]["remember"][0].update(anchor="sofa"), "remember.anchor: 'sofa' is not an available"),
    (lambda d: d["shots"][0].pop("description"), 'shot 1 needs a "title" and a "description"'),
    (lambda d: d.update(shots=[]), '"shots" must be a non-empty list'),
    (lambda d: d.update(shots=d["shots"] * 3), "too many shots: 9, the limit is 6"),
])
def test_what_the_loops_could_not_run_is_rejected_with_a_reason(change, expected):
    plan, errors = to_plan(edited(change), "x", ROLES, 6)
    assert plan is None and any(expected in e for e in errors), errors


def test_text_from_the_model_cannot_break_the_prompt_templates():
    """Hint text later goes through str.format: braces written by the model must not crash the fast loop."""
    plan, errors = to_plan(edited(lambda d: d["shots"][1].update(notes=["Look {off camera}"])), "x", ROLES, 6)
    assert not errors
    note = plan.shots[1].constraints[3]
    assert degraded_prompt(note, plan.subjects) == "Look (off camera)"


def test_lists_longer_than_the_limits_are_cut_not_rejected():
    plan, errors = to_plan(edited(lambda d: d["shots"][0].update(notes=["a", "b", "c", "d"])), "x", ROLES, 6)
    assert not errors and sum(c.checker is Checker.NONE for c in plan.shots[0].constraints) == 2


def test_json_is_found_inside_fences_and_reasoning():
    wrapped = "<think>let me plan {a}</think>\n```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```\nDone."
    make, _ = generator(wrapped)
    assert make("x").style == "The bag left under the table"


# ───────────── Validation failed: ask once more, with the errors ─────────────

def test_a_rejected_list_goes_back_to_the_model_with_the_reasons():
    bad = edited(lambda d: d["shots"][0]["checks"][0]["args"].update(a="gun"))
    make, model = generator(bad, GOOD)
    plan = make("x")

    assert plan.style == "The bag left under the table" and len(model.calls) == 2
    retry = model.calls[1]
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert json.loads(retry[2]["content"]) == bad                               # the model sees what it wrote last time
    assert "'gun' is not an available subject id" in retry[3]["content"]


def test_it_gives_up_after_the_configured_number_of_attempts():
    make, model = generator("I cannot help with that.", "still not JSON")
    with pytest.raises(PlanError, match="asked 2 times.*no JSON object"):
        make("x")
    assert len(model.calls) == 2 and make.last_reply == "still not JSON"


def test_a_model_that_cannot_be_reached_is_not_asked_again():
    calls = []

    def down(messages):
        calls.append(1)
        raise LLMError("Cannot reach the model")

    with pytest.raises(PlanError, match="Cannot reach the model"):
        PlanGenerator(down, ROLES)("x")
    assert len(calls) == 1


def test_an_empty_request_never_reaches_the_model():
    make, model = generator(GOOD)
    with pytest.raises(PlanError):
        make("   ")
    assert not model.calls


def test_roles_can_be_narrowed_to_what_the_current_detector_knows():
    with_cup = edited(lambda d: d["shots"][0]["checks"][0]["args"].update(a="cup"))
    make, model = generator(with_cup, GOOD)
    make("x", roles=frozenset({"person", "bag", "table"}))

    assert "\n  cup\n" not in model.calls[0][0]["content"]                       # the cup is not on the menu at all
    assert "'cup' is not an available subject id" in model.calls[1][3]["content"]   # and using it anyway is sent back


# ───────────── LLM client ─────────────

def chat(handler) -> ChatLLM:
    return ChatLLM("https://api.example/v1/", "sk-secret", "gpt-5.6-luna",
                   client=httpx.Client(transport=httpx.MockTransport(handler)))


def answer(content: str = "{}") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def test_the_request_is_a_minimal_openai_style_chat_completion():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(url=str(request.url), auth=request.headers["authorization"], body=json.loads(request.content))
        return answer('{"ok": true}')

    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert chat(handler).complete(messages) == '{"ok": true}'
    assert seen["url"] == "https://api.example/v1/chat/completions" and seen["auth"] == "Bearer sk-secret"
    assert seen["body"] == {"model": "gpt-5.6-luna", "messages": messages, "response_format": {"type": "json_object"}}


def test_json_mode_is_dropped_for_endpoints_that_refuse_it():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return answer("fine")

    llm = chat(handler)
    assert llm.complete([{"role": "user", "content": "u"}]) == "fine"
    assert llm.complete([{"role": "user", "content": "u"}]) == "fine"
    assert ["response_format" in b for b in bodies] == [True, False, False]     # dropped for good after one refusal


def test_a_refusal_reports_the_servers_reason_and_never_the_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    with pytest.raises(LLMError) as caught:
        chat(handler).complete([{"role": "user", "content": "u"}])
    assert "HTTP 401" in str(caught.value) and "Incorrect API key provided" in str(caught.value)
    assert "sk-secret" not in str(caught.value)


def test_an_empty_or_malformed_reply_is_an_error():
    with pytest.raises(LLMError, match="empty reply"):
        chat(lambda request: answer("  ")).complete([])
    with pytest.raises(LLMError, match="not in the expected format"):
        chat(lambda request: httpx.Response(200, json={"unexpected": True})).complete([])


def test_network_failure_is_an_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(LLMError, match="Cannot reach the model"):
        chat(handler).complete([])


# ───────────── Settings ─────────────

def test_without_a_key_there_is_no_generator():
    assert build_generator(Settings()) is None


def test_settings_read_the_openai_variables_and_the_project_ones_win(monkeypatch):
    for name in ("SHOTAGENT_LLM_API_KEY", "SHOTAGENT_LLM_BASE_URL", "SHOTAGENT_LLM_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    settings = Settings.from_env()
    assert (settings.llm_api_key, settings.llm_base_url, settings.llm_model) == (
        "sk-openai", "https://api.openai.com/v1", "gpt-5.6-luna")
    assert build_generator(settings).name == "gpt-5.6-luna"

    monkeypatch.setenv("SHOTAGENT_LLM_API_KEY", "wandb-key")
    monkeypatch.setenv("SHOTAGENT_LLM_BASE_URL", "https://inference.example/v1")
    monkeypatch.setenv("SHOTAGENT_LLM_MODEL", "some/open-model")
    settings = Settings.from_env()
    assert (settings.llm_api_key, settings.llm_base_url, settings.llm_model) == (
        "wandb-key", "https://inference.example/v1", "some/open-model")
