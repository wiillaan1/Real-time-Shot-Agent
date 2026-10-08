"""Try "idea / script -> shot list" once by itself, without the server: run this the first time you connect a model.

    export OPENAI_API_KEY=sk-...
    python scripts/try_plan.py "Someone working late goes to get water and comes back to find a book on the desk that is not theirs"
    python scripts/try_plan.py --file my_script.txt        # a finished script
    python scripts/try_plan.py --roles person,bag,table "..."  # only these roles (the three the simulated picture has)
    python scripts/try_plan.py --show-prompt "..."            # also print the prompt sent to the model

The default model is gpt-5.6-luna; change model or provider with the environment variables SHOTAGENT_LLM_MODEL /
SHOTAGENT_LLM_BASE_URL / SHOTAGENT_LLM_API_KEY (the same three for the organisers' model on site).

It prints how long it took and who checks each constraint of each shot; on a validation failure, the reasons and the model's raw reply.
It writes no ledger and does not touch data/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shotagent.config import Settings  # noqa: E402
from shotagent.plan_gen import PlanError, system_prompt  # noqa: E402
from shotagent.wiring import build_generator  # noqa: E402

CHECKER = {"yolo": "YOLO, automatic", "pose": "pose, automatic", "sensor": "sensor (text prompt only without a gyro)",
           "cosmos": "semantic model, after the take", "none": "text prompt only"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Try once: idea / script -> shot list")
    parser.add_argument("request", nargs="?", help="an idea, or a script")
    parser.add_argument("--file", type=Path, help="read the script from a file")
    parser.add_argument("--roles", help="only these roles may be used, comma separated")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    request = args.file.read_text(encoding="utf-8") if args.file else (args.request or "")
    if not request.strip():
        parser.error("say what you want to film, or give a script with --file")

    settings = Settings.from_env()
    generator = build_generator(settings)
    if generator is None:
        print("No key: export OPENAI_API_KEY=... first (or SHOTAGENT_LLM_API_KEY)")
        return 2
    roles = frozenset(r.strip() for r in args.roles.split(",") if r.strip()) if args.roles else None
    print(f"Model: {generator.name}   endpoint: {settings.llm_base_url}")
    if args.show_prompt:
        menu_roles = {r: n for r, n in generator._roles.items() if roles is None or r in roles}
        print("--- prompt ---\n" + system_prompt(menu_roles, settings.plan_max_shots) + "\n")

    started = time.perf_counter()
    try:
        plan = generator(request, roles)
    except PlanError as exc:
        print(f"Failed ({time.perf_counter() - started:.1f} s): {exc}")
        if generator.last_reply:
            print("--- the model's raw reply ---\n" + generator.last_reply)
        return 1

    print(f"Done in {time.perf_counter() - started:.1f} s\n")
    print(f"\"{plan.style}\"   {plan.synopsis}")
    print("Subjects: " + ", ".join(f"{name} ({role})" for role, name in plan.subjects.items()))
    for shot in plan.shots:
        print(f"\n{shot.id} \"{shot.title}\"   {shot.setup}   {shot.technique}")
        print(f"  what: {shot.description}")
        print(f"  why:  {shot.intent}")
        for c in shot.constraints:
            print(f"  - {c.text}   [{CHECKER[c.checker.value]}]")
        for e in shot.establishes:
            print(f"  * remembered after a pass: where the {plan.subjects.get(e.subject, e.subject)} is relative to the {plan.subjects.get(e.anchor, e.anchor)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
