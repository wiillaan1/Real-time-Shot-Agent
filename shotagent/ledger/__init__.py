"""The shot ledger.

    state.py    what the ledger looks like (the read model): shot log + scene state. Anyone may import it.
    events.py   the only three kinds of event that can change the ledger.
    reducer.py  the rules: state + event -> new state. Which take is kept and which source a field trusts are all in this one file.
    store.py    Ledger: the single write path commit(), the read-only snapshot(), JSON on disk.

Readers import ledger.state only; writers (planner and slow_loop, nobody else) also import
ledger.store and ledger.events. tests/test_architecture.py enforces this.

This __init__ exports nothing on purpose, so that importing the ledger package never amounts to holding the write path.
"""
