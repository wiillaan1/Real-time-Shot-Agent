"""The semantic judge interface: one take + a few yes/no questions in, verdicts + a semantic description out.

The slow loop knows only this interface. There are two implementations:
    mock.py     returns pass / fail (use it until the organisers' environment is confirmed)
    cosmos.py   a Cosmos3-Reason client written to the organisers' documented call

Dependencies: contracts only.
"""
from __future__ import annotations

from typing import Protocol

from ..contracts import JudgeRequest, JudgeVerdict


class JudgeError(Exception):
    """The semantic judgement could not be made (network down, reply unparseable). The slow loop then leaves the ledger alone."""


class Judge(Protocol):
    name: str    # shown in the UI's status line

    def judge(self, request: JudgeRequest) -> JudgeVerdict:
        """Synchronous, may take seconds. verdict.answers must cover every entry in request.questions."""
        ...
