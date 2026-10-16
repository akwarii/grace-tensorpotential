"""Edit-lock of the tolerance table and the committed golden files: a sha256 manifest checked by the tests.

``LOCK.sha256`` (the format of ``sha256sum``, paths relative to the repository root) holds the digest of the table
(``tolerances.py``) and of every committed golden file: the anchors of ``tests_torch/golden`` (``.npz``, manifests,
yamls) and the manifests and yamls of the faithful tier (``tests_torch/fixtures``; its ``.npz`` files stay local). A
changed, added or removed file fails ``tests_torch/test_tolerances.py`` until the lock is recorded again::

    python -m tests_torch.golden.lock record    # in a commit of its own, whose message says what changed and why
    python -m tests_torch.golden.lock check     # what the test checks; exit status 1 on a difference

so a tolerance or a golden never changes as a side effect of another change.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
LOCK: Final = Path(__file__).resolve().with_name("LOCK.sha256")
PATTERNS: Final = (
    "tests_torch/golden/tolerances.py",
    "tests_torch/golden/*.npz",
    "tests_torch/golden/*.json",
    "tests_torch/golden/yamls/*.yaml",
    "tests_torch/fixtures/*.json",
    "tests_torch/fixtures/yamls/*.yaml",
)
"""The locked files, as globs relative to the repository root."""


def locked_files(root: Path = ROOT) -> list[Path]:
    """Return the files the lock covers under ``root``, sorted."""
    return sorted({
        p for pattern in PATTERNS for p in root.glob(pattern) if p.is_file()
    })


def digests(root: Path = ROOT) -> dict[str, str]:
    """Return the sha256 of every locked file, keyed by its path relative to ``root`` (``/`` separators)."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in locked_files(root)
    }


def format_lock(entries: dict[str, str]) -> str:
    """Return the text of a lock file: one ``<sha256>  <path>`` line per file, sorted by path."""
    return "".join(f"{digest}  {path}\n" for path, digest in sorted(entries.items()))


def read_lock(lock: Path = LOCK) -> dict[str, str]:
    """Read a lock file into ``{path: sha256}``."""
    digest_and_path = (
        line.split("  ", 1) for line in lock.read_text().splitlines() if line
    )
    return {path: digest for digest, path in digest_and_path}


def differences(recorded: dict[str, str], current: dict[str, str]) -> list[str]:
    """Return one line per file that was changed, added or removed since the lock was recorded."""
    changed = [
        f"changed: {p}"
        for p in sorted(recorded.keys() & current.keys())
        if recorded[p] != current[p]
    ]
    added = [f"not in the lock: {p}" for p in sorted(current.keys() - recorded.keys())]
    removed = [f"missing: {p}" for p in sorted(recorded.keys() - current.keys())]
    return changed + added + removed


def main(argv: list[str], root: Path = ROOT, lock: Path = LOCK) -> int:
    """Run ``record`` or ``check`` on the files under ``root``; return the exit status."""
    if argv == ["record"]:
        lock.write_text(format_lock(digests(root)))
        return 0
    if argv == ["check"]:
        found = differences(read_lock(lock), digests(root))
        sys.stdout.write("".join(f"{line}\n" for line in found))
        return 1 if found else 0
    sys.stderr.write("usage: python -m tests_torch.golden.lock record|check\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
