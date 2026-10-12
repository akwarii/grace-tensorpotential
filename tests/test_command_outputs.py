"""What ``grace_utils export``, ``export_kokkos`` and ``grace_preprocess`` write must not change when their code moves.

The reference ``tests/data/command_outputs.json`` was recorded on the tree before CLI2 moved any code (see
``tests/command_outputs.py`` for how an output is digested). The inputs are seeded, so the outputs are reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.command_outputs import OUTPUTS, digest_tree

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "command_outputs.json").read_text()
)


def test_the_reference_covers_every_command_output():
    assert set(REFERENCE) == set(OUTPUTS)


@pytest.mark.parametrize("name", sorted(OUTPUTS))
def test_the_command_writes_the_files_and_the_content_of_the_reference(name, tmp_path):
    out = OUTPUTS[name](tmp_path)
    assert digest_tree(out) == REFERENCE[name]
