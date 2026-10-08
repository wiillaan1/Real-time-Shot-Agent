"""Idea or script -> shot list (Plan): the step of the planner that uses an LLM.

    PlanGenerator(complete, roles)(request) -> Plan

The user may give a one-line idea ("someone working late finds a coffee on the desk that is not theirs"), a style,
or a finished script. The model's job is to break it into shots and give each shot constraints the system can **actually check**.

The hard part is not getting the model to write a story, it is the constraints: the fast loop only knows the relations registered
in constraints.PREDICATES, and the detector only recognises the roles in config.DEFAULT_LABEL_MAP. So this file does three things:

  1. Menu       Put "which roles can be seen, which relations can be checked" into the prompt. The menu is computed from PREDICATES
                and the role table, not copied by hand: register a new predicate and the model knows about it next time.
  2. Routing    Whatever the menu cannot express, the model must put into semantic (a yes/no question the semantic model answers after the take)
                or notes (a text prompt nobody checks). This is the design doc's "label it honestly" (2.2).
  3. Validation The model's output is never used as a Plan directly: predicate and role names are checked one by one, numbers are clamped,
                and IDs, checkers and default wording are filled in here. If validation fails the errors go back to the model for one rewrite.

The model only decides what to film and what to check; it never touches IDs or wording templates, so there is little it can break.

This file does not know who the model is: complete is a "conversation in, text out" function (wiring.py plugs in llm.ChatLLM).

Dependencies: contracts, constraints (only for the PREDICATES table).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from .constraints import PREDICATES, Predicate
from .contracts import Checker, Constraint, Establish, Loop, Plan, Shot

Complete = Callable[[Sequence[dict[str, str]]], str]

MAX_CHECKS, MAX_SEMANTIC, MAX_NOTES, MAX_REMEMBER = 4, 2, 2, 2
MAX_REQUEST_CHARS = 4000
DEFAULT_SETUP = "Setup A"


class PlanError(Exception):
    """The shot list was not generated (no model configured, the call failed, or the list failed validation twice). The ledger is not changed."""


# ───────────────────────── Prompt ─────────────────────────

def _signature(name: str, spec: Predicate) -> str:
    parts = list(spec.roles)
    parts += [f"{key}={default:g} [{low:g}..{high:g}]" for key, (default, low, high) in spec.numbers.items()]
    limits = "".join(f"; {key} must be {' or '.join(allowed)}" for key, allowed in spec.only.items())
    return f"  {name}({', '.join(parts)}): {spec.meaning}{limits}"


def menu(roles: Mapping[str, str]) -> str:
    """The menu shown to the model: the roles it can see + the relations that can be checked automatically."""
    subjects = "\n".join(f"  {role}" + (f" ({name})" if name != role else "") for role, name in roles.items())
    predicates = "\n".join(_signature(name, spec) for name, spec in PREDICATES.items())
    return (
        "Subjects the software can see. Use only these ids, and at most one of each:\n"
        f"{subjects}\n\n"
        "Automatic checks (\"predicates\"), evaluated on every frame while the shot is being framed. "
        "Only these exist:\n"
        f"{predicates}"
    )


def system_prompt(roles: Mapping[str, str], max_shots: int) -> str:
    return f"""You are the planning layer of a live shooting assistant. One person films a short piece with a single camera, and software supervises every shot while it is being framed. Turn the user's request into a shot list that the software can supervise.

The request is either a rough idea (a premise, a mood, a director's style) or a finished script.
- An idea: invent a very small story that fits it.
- A script: keep its beats and its wording; do not add plot.
Either way, design for a crew of one or two people in one room: at most one person on screen, a few props, between 3 and {max_shots} shots, each shot about five seconds long.

{menu(roles)}

Prefer stories that need only two or three subjects besides the person. Other things may appear in the story, but the software cannot see them.

Whatever the predicates cannot express goes into one of two places:
- "semantic": a claim about the finished take that a video model answers yes or no (for example "the viewer can tell the cup is not hers").
- "notes": a reminder shown to the crew that nobody checks (acting, lighting, timing).
Do not invent predicates and do not bend one to mean something else.

"remember": when a shot passes, the software can record where one subject sits relative to another (left or right side, under or above), for example {{"subject": "bag", "anchor": "table"}}. Later shots are then guided from that record ("the bag is on the left, so keep the person on the right") even when the subject is hidden. Both subjects must be in the frame in the shot that remembers them. Use it whenever a later shot uses apart or not_facing against that subject.

Reply with one JSON object and nothing else:
{{
  "title": "short title of the piece",
  "synopsis": "the story in two or three sentences",
  "subjects": {{"<subject id>": "<what it is called in this story>"}},
  "shots": [
    {{
      "title": "short name of the shot",
      "description": "what to film: who and what is in the frame, and what happens",
      "intent": "why the story needs this shot",
      "technique": "name of the film technique, if any",
      "setup": "camera position, e.g. Setup A (table, front, medium shot). Reuse the exact same text for shots filmed from the same position and keep those shots next to each other",
      "checks": [{{"predicate": "apart", "args": {{"a": "person", "b": "bag", "ruler": "table", "min_ratio": 0.5}}}}],
      "semantic": [{{"text": "the claim as a statement", "ask": "the same claim as a yes/no question"}}],
      "notes": ["a reminder for the crew"],
      "remember": [{{"subject": "bag", "anchor": "table"}}]
    }}
  ]
}}
Per shot: at most {MAX_CHECKS} checks, {MAX_SEMANTIC} semantic claims, {MAX_NOTES} notes. Empty lists are fine.
Write everything in English."""


# ───────────────────────── Model reply -> Plan ─────────────────────────

def extract_json(reply: str) -> Any:
    """Find the JSON in a reply: tolerates <think>...</think>, ```json fences and chatter before or after."""
    text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the reply contains no JSON object")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"the reply is not valid JSON ({exc.msg} at character {exc.pos})") from exc


def _text(value: Any, limit: int = 200) -> str:
    """A piece of text written by the model: squeezed to one line and capped in length.
    Braces are replaced because this text later goes through str.format in the hint templates."""
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(value.split()).replace("{", "(").replace("}", ")")
    return cleaned[:limit]


def _items(value: Any, limit: int) -> list:
    return list(value)[:limit] if isinstance(value, list) else []


def to_plan(data: Any, request: str, roles: Mapping[str, str], max_shots: int) -> tuple[Optional[Plan], list[str]]:
    """Validate the model's list and convert it to a Plan. Returns (Plan or None, errors); the errors are written for the model."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return None, ["the reply must be a JSON object"]
    raw_shots = data.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        return None, ['"shots" must be a non-empty list']
    if len(raw_shots) > max_shots:
        errors.append(f"too many shots: {len(raw_shots)}, the limit is {max_shots}")

    given_names = data.get("subjects") if isinstance(data.get("subjects"), dict) else {}
    used: set[str] = set()

    def display(role: str) -> str:
        return _text(given_names.get(role), 40) or roles[role]

    def role_of(value: Any, where: str) -> Optional[str]:
        if not isinstance(value, str) or value not in roles:
            errors.append(f'{where}: {value!r} is not an available subject id (available: {", ".join(roles)})')
            return None
        used.add(value)
        return value

    shots: list[Shot] = []
    for index, raw in enumerate(raw_shots[:max_shots], start=1):
        shot_id, where = f"S{index:02d}", f"shot {index}"
        if not isinstance(raw, dict):
            errors.append(f"{where} must be an object")
            continue
        title, description = _text(raw.get("title"), 80), _text(raw.get("description"), 400)
        if not title or not description:
            errors.append(f'{where} needs a "title" and a "description"')

        checks: list[Constraint] = []
        for raw_check in _items(raw.get("checks"), MAX_CHECKS):
            name = raw_check.get("predicate") if isinstance(raw_check, dict) else None
            spec = PREDICATES.get(name) if isinstance(name, str) else None
            if spec is None:
                errors.append(f'{where}: unknown predicate {name!r} (available: {", ".join(PREDICATES)})')
                continue
            raw_args = raw_check.get("args") if isinstance(raw_check.get("args"), dict) else {}
            args: dict[str, Any] = {}
            valid = True
            for key in spec.roles:
                role = role_of(raw_args.get(key), f"{where} {name}.{key}")
                if role is None:
                    valid = False
                elif key in spec.only and role not in spec.only[key]:
                    errors.append(f'{where} {name}.{key} must be {" or ".join(spec.only[key])}, not "{role}"')
                    valid = False
                args[key] = role
            if valid and len(set(args.values())) != len(args):
                errors.append(f"{where} {name}: its arguments must be different subjects")
                valid = False
            if not valid:
                continue
            for key, (default, low, high) in spec.numbers.items():
                try:
                    number = float(raw_args.get(key, default))
                except (TypeError, ValueError):
                    number = default
                args[key] = min(max(number, low), high)   # numbers are not sent back for a rewrite, just clamped into range
            slots = {key: display(value) if key in spec.roles else f"{value:g}" for key, value in args.items()}
            checks.append(Constraint(
                id="", text=spec.text.format_map(slots), checker=spec.checker, loop=Loop.FAST,
                predicate=name, args=args, needs=spec.needs, fix=spec.fix, prompt=spec.prompt,
            ))

        notes = [
            Constraint(id="", text=note, checker=Checker.NONE, loop=Loop.FAST, prompt=note)
            for note in (_text(item) for item in _items(raw.get("notes"), MAX_NOTES)) if note
        ]

        semantic: list[Constraint] = []
        for item in _items(raw.get("semantic"), MAX_SEMANTIC):
            claim = _text(item.get("text")) if isinstance(item, dict) else _text(item)
            if claim:
                ask = _text(item.get("ask"), 300) if isinstance(item, dict) else ""
                semantic.append(Constraint(id="", text=claim, checker=Checker.COSMOS, loop=Loop.SLOW, ask=ask or claim))

        establishes: list[Establish] = []
        for item in _items(raw.get("remember"), MAX_REMEMBER):
            if not isinstance(item, dict):
                continue
            subject = role_of(item.get("subject"), f"{where} remember.subject")
            anchor = role_of(item.get("anchor"), f"{where} remember.anchor")
            if subject and anchor and subject != anchor:
                establishes.append(Establish(subject=subject, anchor=anchor))
            elif subject and anchor:
                errors.append(f"{where} remember: subject and anchor must be different subjects")

        # The order is the hint priority: automatically checkable first, then text prompts, then the ones judged after the take
        ordered = checks + notes + semantic
        shots.append(Shot(
            id=shot_id, order=index, title=title, description=description,
            intent=_text(raw.get("intent"), 400), technique=_text(raw.get("technique"), 80),
            setup=_text(raw.get("setup"), 80) or DEFAULT_SETUP,
            constraints=tuple(c.model_copy(update={"id": f"{shot_id}.c{n}"}) for n, c in enumerate(ordered, start=1)),
            establishes=tuple(establishes),
        ))

    if errors:
        return None, errors
    return Plan(
        style=_text(data.get("title"), 80) or _text(request, 40),
        subjects={role: display(role) for role in roles if role in used},
        shots=tuple(shots), request=request, synopsis=_text(data.get("synopsis"), 600),
    ), []


# ───────────────────────── Generator ─────────────────────────

class PlanGenerator:
    def __init__(self, complete: Complete, roles: Mapping[str, str], *, max_shots: int = 6,
                 attempts: int = 2, name: str = ""):
        """roles: role name -> default display name, i.e. every role the model may use. name: model name, shown in the UI."""
        self._complete = complete
        self._roles = dict(roles)
        self._max_shots = max_shots
        self._attempts = max(attempts, 1)
        self.name = name
        self.last_reply = ""    # the model's last raw reply, for tuning the prompt

    def __call__(self, request: str, roles: Optional[frozenset[str]] = None) -> Plan:
        """roles: only these roles may be used this time (the ones the current detector recognises); None = all."""
        request = request.strip()[:MAX_REQUEST_CHARS]
        if not request:
            raise PlanError("Nothing was entered about what to film")
        allowed = {r: n for r, n in self._roles.items() if roles is None or r in roles} or self._roles

        messages = [
            {"role": "system", "content": system_prompt(allowed, self._max_shots)},
            {"role": "user", "content": request},
        ]
        errors: list[str] = []
        for _ in range(self._attempts):
            try:
                reply = self._complete(messages)
            except Exception as exc:   # a failed call is not retried: it is almost always the key, the network or the model name, and asking again changes nothing
                raise PlanError(str(exc)) from exc
            self.last_reply = reply
            try:
                plan, errors = to_plan(extract_json(reply), request, allowed, self._max_shots)
            except ValueError as exc:
                plan, errors = None, [str(exc)]
            if plan is not None:
                return plan
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "That shot list cannot be used:\n"
                    + "\n".join(f"- {error}" for error in errors)
                    + "\nSend the complete corrected JSON object."},
            ]
        shown = "; ".join(errors[:3]) + (" ..." if len(errors) > 3 else "")
        raise PlanError(f"The model's shot list failed validation (asked {self._attempts} times): {shown}")
