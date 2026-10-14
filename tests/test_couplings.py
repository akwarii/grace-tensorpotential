"""``functions/couplings.py::real_coupling_metainformation``: coupling of two (l, m, history) tables.

Logic layer: every filter (``lmax*``, ``max_sum_l``, ``keep_parity``, ``history_drop_list``,
``is_A_B_equal``), the legacy and parity-marked history strings, and the typed errors.
Exact layer: a SHA-256 digest of the complete result (every column, every float to its last
bit) for the tables the presets build and for a grid of option combinations; the digests in
``tests/data/coupling_metainformation_digests.json`` were taken before the function was
split into steps (#177), so the split must reproduce them bit for bit.
Value layer: closed-form real Clebsch-Gordan coefficients (``1/sqrt(3)`` for the scalar
product of two vectors, ``1/sqrt(2)`` pairs for their cross product).

Regenerate the digests (only after an intended change of the coupling tables) with
``python tests/test_couplings.py``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import pathlib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import pytest

from tensorpotential.functions.couplings import real_coupling_metainformation
from tensorpotential.potentials import get_preset

DIGESTS = (
    pathlib.Path(__file__).parent / "data" / "coupling_metainformation_digests.json"
)
CG_ATOL = 1e-12  # closed forms are exact to double precision

PRESET_CASES = [
    ("LINEAR", {"lmax": 2}),
    ("FS", {"lmax": 2}),
    ("GRACE_1LAYER_v1_24", {"lmax": 1}),
    ("GRACE_1LAYER_v1_24", {"lmax": 3}),
    ("GRACE_2LAYER_v1_24", {"lmax": 2}),
    ("GRACE_1LAYER_latest", {"lmax": 1}),
    ("GRACE_1LAYER_latest", {"lmax": 3}),
    (
        "GRACE_2LAYER_latest",
        {
            "lmax": [3, 2],
            "n_rad_max": [4, 4],
            "prod_func_n_max": [4, 4],
            "indicator_lmax": 1,
            "max_order": 3,
        },
    ),
]
BASE_CONFIG = {"element_map": {"H": 0, "O": 1}, "rcut": 4.5, "n_rad_max": 4}
GRID_OPTIONS = {
    "is_A_B_equal": [False, True],
    "normalize": [False, True],
    "optimize_ms_comb": [True, False],
    "legacy_format": [False, True],
    "keep_parity": [None, [[0, 1], [1, -1], [2, 1]]],
    "max_sum_l": [None, 4],
    "lmax_hist": [None, 1],
    "lmax_A": [None, 1],
    "history_drop_list": [None, [("(0,0)0", 0)]],
}
GRID_STEP = 7  # every 7th of the 512 combinations: 74 per frame, three frames, all values of every option


def _canonical(value: object) -> object:
    """Turn a table cell into JSON-able data that keeps every float bit."""
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_canonical(v) for v in value]
    if isinstance(value, (np.integer, int, str)):
        return value if isinstance(value, str) else int(value)
    return str(value)  # symbolic columns (``symbol``) of the harmonics tables


def digest(frame: pd.DataFrame) -> str:
    """SHA-256 of the columns, the row order and every cell of ``frame``."""
    sha = hashlib.sha256()
    for column in frame.columns:
        sha.update(
            json.dumps([column, [_canonical(v) for v in frame[column]]]).encode()
        )
    return sha.hexdigest()


def _preset_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for name, extra in PRESET_CASES:
        config = {**BASE_CONFIG, **extra}
        instructions = get_preset(name)(**config).get_instructions()
        for ins_name, instruction in instructions.items():
            frame = getattr(instruction, "coupling_meta_data", None)
            if frame is not None:
                frames[f"{name}:{extra['lmax']}:{ins_name}"] = frame
    return frames


def _grid_cases(frames: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """A deterministic subset of the option grid on three frames of different size."""
    chosen = {
        "A": frames["GRACE_1LAYER_latest:1:A"],
        "AA": frames["GRACE_1LAYER_latest:3:AA"],
        "Y3": frames["GRACE_1LAYER_latest:3:Y"],
    }
    keys = list(GRID_OPTIONS)
    combos = [
        dict(zip(keys, values))
        for values in itertools.product(*(GRID_OPTIONS[k] for k in keys))
    ]
    cases = {}
    for frame_name, frame in chosen.items():
        for i, options in enumerate(combos[::GRID_STEP]):
            kwargs = {"A": frame, "B": frame, "lmax": 2, "Lmax": 2, **options}
            cases[f"grid:{frame_name}:{i}"] = kwargs
    return cases


def _grid_digest(kwargs: dict) -> str:
    try:
        return digest(real_coupling_metainformation(**kwargs))
    except (
        ValueError,
        KeyError,
    ) as exc:  # a combination may be rejected: pin the error
        return f"{type(exc).__name__}: {exc}"


def _all_digests() -> dict[str, str]:
    frames = _preset_frames()
    result = {name: digest(frame) for name, frame in frames.items()}
    for name, kwargs in _grid_cases(frames).items():
        result[name] = _grid_digest(kwargs)
    return result


@pytest.fixture(scope="module")
def current_digests() -> dict[str, str]:
    return _all_digests()


@pytest.fixture(scope="module")
def reference_digests() -> dict[str, str]:
    return json.loads(DIGESTS.read_text())


def test_the_digests_cover_every_preset_instruction_and_the_grid(
    reference_digests, current_digests
):
    assert set(current_digests) == set(reference_digests)
    assert len(reference_digests) == 306  # 84 preset tables and 222 grid cases


def test_every_table_is_bit_identical_to_the_reference(
    reference_digests, current_digests
):
    wrong = sorted(k for k, v in reference_digests.items() if current_digests[k] != v)
    assert not wrong, f"{len(wrong)} tables changed, first: {wrong[:5]}"


def _frame(rows: list[tuple[int, int, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["l", "m", "hist", "parity", "sum_of_ls"])


VECTOR = _frame([(1, -1, "1", -1, 1), (1, 0, "1", -1, 1), (1, 1, "1", -1, 1)])
SCALAR = _frame([(0, 0, "0", 1, 0)])


def test_scalar_product_of_two_vectors_is_the_trace_over_root_three():
    # real harmonics: Y_1 . Y_1 -> L=0 with cg = 1/sqrt(3) on each diagonal pair
    result = real_coupling_metainformation(
        A=VECTOR, B=VECTOR, lmax=1, Lmax=0, normalize=False
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert (row["l"], row["m"], row["parity"]) == (0, 0, 1)
    assert sorted(zip(row["left_inds"], row["right_inds"])) == [(0, 0), (1, 1), (2, 2)]
    np.testing.assert_allclose(row["cg_list"], 1 / math.sqrt(3), rtol=0, atol=CG_ATOL)


def test_cross_product_of_two_vectors_has_two_terms_of_one_over_root_two_per_component():
    result = real_coupling_metainformation(
        A=VECTOR, B=VECTOR, lmax=1, Lmax=1, normalize=False
    )
    cross = result[result["l"] == 1]
    assert len(cross) == 3
    for cg_list, left in zip(cross["cg_list"], cross["left_inds"]):
        assert len(left) == 2
        np.testing.assert_allclose(
            sorted(np.abs(cg_list)), [1 / math.sqrt(2)] * 2, rtol=0, atol=CG_ATOL
        )
        assert cg_list[0] == pytest.approx(-cg_list[1], abs=CG_ATOL)


def test_equal_operands_keep_the_symmetric_part_only_and_halve_the_pairs():
    plain = real_coupling_metainformation(A=VECTOR, B=VECTOR, lmax=1, Lmax=2)
    equal = real_coupling_metainformation(
        A=VECTOR, B=VECTOR, lmax=1, Lmax=2, is_A_B_equal=True
    )
    assert set(equal["l"]) == {0, 2}  # the antisymmetric L=1 combination cancels
    assert len(equal) < len(plain)


def test_history_strings_carry_parity_markers_unless_legacy():
    marked = real_coupling_metainformation(A=VECTOR, B=SCALAR, lmax=1, Lmax=1)
    legacy = real_coupling_metainformation(
        A=VECTOR, B=SCALAR, lmax=1, Lmax=1, legacy_format=True
    )
    assert set(marked["hist"]) == {"(11-,00+)"}
    assert set(legacy["hist"]) == {"(11,00)"}


@pytest.mark.parametrize(
    ("kwargs", "expected_l"),
    [
        ({"lmax": 0}, set()),
        ({"lmax_A": 0}, set()),
        ({"lmax_B": 0, "lmax": 1}, {1}),
        ({"max_sum_l": 0}, set()),
        ({"lmax_hist": 0}, set()),
        ({"lmax_hist_A": 0}, set()),
        ({"lmax_hist_B": 0}, {1}),
        ({"keep_parity": [[1, -1]]}, {1}),
        ({"keep_parity": [[1, 1]]}, set()),
        ({"Lmax": 0}, set()),
    ],
)
def test_each_filter_removes_the_couplings_it_excludes(kwargs, expected_l):
    options = {"lmax": 1, "Lmax": 1, **kwargs}
    result = real_coupling_metainformation(A=VECTOR, B=SCALAR, **options)
    assert set(result["l"]) == expected_l


def test_keep_parity_accepts_a_list_of_lists_and_rejects_other_containers():
    with pytest.raises(ValueError, match="Only list of"):
        real_coupling_metainformation(
            A=VECTOR, B=SCALAR, lmax=1, Lmax=1, keep_parity=((1, -1),)
        )


@pytest.mark.parametrize("side", ["A", "B"])
def test_a_table_with_a_repeated_l_m_history_is_rejected(side):
    twice = pd.concat([VECTOR, VECTOR.iloc[:1]], ignore_index=True)
    tables = {"A": VECTOR, "B": VECTOR, side: twice}
    with pytest.raises(ValueError, match=f"{side} has non-unique"):
        real_coupling_metainformation(**tables, lmax=1, Lmax=1)


def test_history_drop_list_removes_a_history_for_the_given_target_l_only():
    full = real_coupling_metainformation(
        A=VECTOR, B=SCALAR, lmax=1, Lmax=1, legacy_format=True
    )
    dropped = real_coupling_metainformation(
        A=VECTOR,
        B=SCALAR,
        lmax=1,
        Lmax=1,
        legacy_format=True,
        history_drop_list=[
            ["(11,00)", 1]
        ],  # list of lists, as restored from a model.yaml
    )
    assert len(full) == 3
    assert dropped.empty
    kept = real_coupling_metainformation(
        A=VECTOR,
        B=SCALAR,
        lmax=1,
        Lmax=1,
        legacy_format=True,
        history_drop_list=[("(11,00)", 0), "(11,00)"],  # other L, entry without L
    )
    assert len(kept) == 3


def test_the_default_exclusion_lists_differ_between_legacy_and_parity_aware_paths():
    kwargs = {"A": VECTOR, "B": VECTOR, "lmax": 1, "Lmax": 2}
    assert len(real_coupling_metainformation(**kwargs, legacy_format=True)) > 0
    assert len(real_coupling_metainformation(**kwargs)) > 0


def test_the_result_is_sorted_by_l_parity_hist_and_m_with_a_fresh_index():
    result = real_coupling_metainformation(A=VECTOR, B=VECTOR, lmax=1, Lmax=2)
    keys = list(zip(result["l"], result["parity"], result["hist"], result["m"]))
    assert keys == sorted(keys)
    assert list(result.index) == list(range(len(result)))


if __name__ == "__main__":
    DIGESTS.write_text(json.dumps(_all_digests(), indent=0, sort_keys=True) + "\n")
    print("wrote", DIGESTS)
