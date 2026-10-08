"""The executable version of the module dependencies.

The "who depends on whom" picture in the README is written here as a table and checked against the real imports on every test run.
Adding a module or a dependency means editing this table first, which forces the question of whether the new relation should exist.
"""
from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "shotagent"

# Module -> the internal modules it may import
ALLOWED: dict[str, set[str]] = {
    # Foundation: no internal dependencies
    "config": set(),
    "contracts": set(),
    "texts": set(),
    "media": set(),
    "llm": set(),
    "spatial": {"config", "contracts"},
    # Ledger
    "ledger": set(),
    "ledger.state": {"contracts"},
    "ledger.events": {"contracts", "ledger.state"},
    "ledger.reducer": {"contracts", "ledger.state", "ledger.events"},
    "ledger.store": {"ledger.state", "ledger.events", "ledger.reducer"},
    # Planning
    "plans": {"contracts"},
    "plan_gen": {"contracts", "constraints"},
    "planner": {"contracts", "plans", "plan_gen", "ledger.store", "ledger.events"},
    # Perception
    "perception": {"config", "perception.base", "perception.null", "perception.yolo"},
    "perception.base": {"contracts"},
    "perception.null": {"contracts"},
    "perception.synthetic": {"contracts"},
    "perception.yolo": {"contracts"},
    # Fast loop
    "constraints": {"config", "contracts", "spatial", "texts", "ledger.state"},
    "fast_loop": {"config", "contracts", "constraints", "spatial", "texts", "perception.base", "ledger.state"},
    # Slow loop
    "judge": {"config", "judge.base", "judge.mock", "judge.cosmos"},
    "judge.base": {"contracts"},
    "judge.mock": {"contracts"},
    "judge.cosmos": {"contracts", "media", "judge.base"},
    "takes": {"contracts", "media"},
    "slow_loop": {"config", "contracts", "constraints", "spatial", "texts", "judge.base",
                  "ledger.state", "ledger.events", "ledger.store"},
    # Tying it together
    "frame_source": {"config", "contracts", "media"},
    "session": {"config", "contracts", "texts", "fast_loop", "slow_loop", "takes", "planner", "ledger.state"},
    "wiring": {"config", "contracts", "fast_loop", "frame_source", "judge", "judge.base", "ledger.store",
               "llm", "perception", "perception.base", "perception.synthetic", "plan_gen", "planner", "session",
               "slow_loop", "takes"},
    "server": {"config", "contracts", "media", "frame_source", "perception.synthetic", "plan_gen", "session",
               "takes", "wiring"},
}

# Only these modules may touch the ledger's write side (the write path + the write events)
WRITE_SIDE = {"ledger.store", "ledger.events"}
MAY_TOUCH_WRITE_SIDE = {"planner", "slow_loop", "wiring", "ledger.store", "ledger.reducer", "ledger.events"}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PACKAGE).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


MODULES = {_module_name(p): p for p in PACKAGE.rglob("*.py") if _module_name(p)}


def _internal_imports(name: str, path: Path) -> set[str]:
    """Which internal modules this module imports (relative imports resolved to module names inside the shotagent package)."""
    is_package = path.name == "__init__.py"
    package_parts = name.split(".") if is_package else name.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            base = package_parts[: len(package_parts) - (node.level - 1)]
            target = base + (node.module.split(".") if node.module else [])
            for alias in node.names:
                as_submodule = ".".join(target + [alias.name])
                found.add(as_submodule if as_submodule in MODULES else ".".join(target))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("shotagent"):
            found.add(".".join(node.module.split(".")[1:]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("shotagent."):
                    found.add(alias.name.split(".", 1)[1])
    found.discard("")
    found.discard(name)
    return found


def test_every_module_has_declared_dependencies():
    assert set(MODULES) == set(ALLOWED), "New module? State what it may depend on in ALLOWED first"


def test_imports_follow_the_declared_dependencies():
    problems = []
    for name, path in sorted(MODULES.items()):
        extra = _internal_imports(name, path) - ALLOWED[name]
        if extra:
            problems.append(f"{name} should not import {sorted(extra)}")
    assert not problems, "\n".join(problems)


def test_only_planner_and_slow_loop_reach_the_ledger_write_side():
    offenders = [
        name for name, path in MODULES.items()
        if name not in MAY_TOUCH_WRITE_SIDE and _internal_imports(name, path) & WRITE_SIDE
    ]
    assert not offenders, f"These modules should not touch the ledger's write side: {offenders}"


def test_fast_loop_cannot_write_the_ledger():
    """The fast loop only reads snapshots: it does not import the write side and holds no Ledger object."""
    from shotagent.fast_loop import FastLoop
    from shotagent.ledger.store import Ledger
    from shotagent.perception.null import NullDetector
    from shotagent.spatial import DEFAULT_TUNING

    assert not _internal_imports("fast_loop", MODULES["fast_loop"]) & WRITE_SIDE
    loop = FastLoop(lambda caps: NullDetector(), DEFAULT_TUNING)
    assert not any(isinstance(value, Ledger) for value in vars(loop).values())


def test_dependency_table_has_no_cycles():
    seen: dict[str, int] = {}

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if seen.get(name) == 2:
            return
        assert seen.get(name) != 1, f"Dependency cycle: {' -> '.join(trail + (name,))}"
        seen[name] = 1
        for dep in ALLOWED[name]:
            visit(dep, trail + (name,))
        seen[name] = 2

    for module in ALLOWED:
        visit(module, ())
