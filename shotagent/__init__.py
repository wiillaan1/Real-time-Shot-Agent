"""Backend of the live shooting agent.

Where to start reading:
    wiring.py      who holds whom: the whole system is assembled in this one function
    session.py     the agent loop itself: framing -> recording -> checking -> next shot
    fast_loop.py   fast loop: one frame in, one framing hint out (reads a ledger snapshot only)
    slow_loop.py   slow loop: one take in, conclusions written to the ledger
    ledger/        the ledger: single write path, rules, state
    contracts.py   what the data passed between these modules looks like

The rest are the parts they use:
    plans.py / planner.py   the three hand-written shots, and the planner that writes a list to the ledger
    plan_gen.py / llm.py    the generator that turns an idea or a script into a shot list, and the model client it uses
    constraints.py          relation predicates ("far enough apart", "not facing it")
    spatial.py              geometry for relative position and facing
    perception/             detectors: YOLO, simulated picture, null
    judge/                  semantic judges: mock, Cosmos
    takes.py / media.py     saving takes, image and video encoding
    frame_source.py         frame intake (video source -> Frame)
    server.py               HTTP endpoints
    config.py / texts.py    tunable parameters, shared user-facing wording

The allowed dependencies between modules are written as a test: tests/test_architecture.py.
"""
