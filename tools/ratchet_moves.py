"""Moved files for the ratchet tools: ``--moved OLD NEW``.

A refactor that moves a file must not reset what the ratchets know about it. ``lint_ratchet.py`` and
``coverage_ratchet.py`` re-key their baseline from ``OLD`` to ``NEW`` before they compare, so the counts
(and the legacy status) follow the file and a rise is still refused at the new path.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def parse_moves(pairs: list[list[str]] | None) -> dict[str, str]:
    """``{old: new}`` from the ``--moved OLD NEW`` pairs; a pair that cannot be a move is an error."""
    moves: dict[str, str] = {}
    for old, new in pairs or []:
        if old == new:
            msg = f"--moved {old} {new}: the old and the new path are the same"
            raise ValueError(msg)
        if old in moves:
            msg = f"--moved: {old} is moved twice"
            raise ValueError(msg)
        moves[old] = new
    return moves


def rekey(table: dict[str, T], moves: dict[str, str]) -> dict[str, T]:
    """``table`` with the key of every moved file replaced by its new path."""
    out: dict[str, T] = {}
    for key, value in table.items():
        new = moves.get(key, key)
        if new in out:
            msg = f"{new} would hold two baseline entries after the moves"
            raise ValueError(msg)
        out[new] = value
    return out
