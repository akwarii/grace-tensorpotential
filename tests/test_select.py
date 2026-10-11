"""``tensorpotential.uq.cli.select.select_main``: the ``grace_uq select`` command line, run in process.

The candidate pools are small synthetic ``grace_uq predict`` outputs (``ase_atoms``, per-atom ``gamma`` and ``features``)
built here from chosen numbers, with the synthetic UQ artifact of ``tests/test_uq_cli.py`` (elements Mo and Nb, D = 4, so the
default stratification floor is 5 atoms per element) or the committed SavedModel ``tests/test_calculator_model`` for the
element map. Nothing needs a TensorFlow model: the prediction step is not run here.

Logic layer: the options, every ``SystemExit`` message, the sources of the element map and of D, the warnings, the output formats.
Physical values layer (hand oracles on the chosen gamma values): the structures selected by ``random-extrap`` are exactly
those with an atom in the gamma window ``[gamma_min, gamma_max]``; a floor of atoms per element holds in the selected
structures; the ``fps-extrap`` picks stay inside the window.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read

from tensorpotential.uq.cli import select as select_mod
from tensorpotential.uq.cli.select import select_main

from .test_uq_cli import _write_synthetic_info_artifact

SAVED_MODEL = Path(__file__).resolve().parent / "test_calculator_model"
N_STRUCTURES = 10
FEATURE_DIM = 4


def pool_frame(n: int = N_STRUCTURES, with_features: bool = True) -> pd.DataFrame:
    """``n`` structures of four atoms (Mo, Mo, Nb, Nb) with gamma ``[0.25, 0.75, 1 + s/4, 2 + s/4]`` for structure ``s``."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n):
        atoms = Atoms(
            ["Mo", "Mo", "Nb", "Nb"],
            positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]],
        )
        row = {
            "ase_atoms": atoms,
            "name": f"s{s}",
            "gamma": np.array([0.25, 0.75, 1.0 + 0.25 * s, 2.0 + 0.25 * s]),
            "energy_predicted": -1.0 * s,
        }
        if with_features:
            row["features"] = rng.normal(size=(4, FEATURE_DIM)).astype(np.float32)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def artifact(tmp_path) -> str:
    path = str(tmp_path / "uq.npz")
    _write_synthetic_info_artifact(path, n_elements=2, K=3, D=FEATURE_DIM)
    return path


def write_pool(
    tmp_path: Path, frame: pd.DataFrame | None = None, name: str = "predicted.pkl.gz"
) -> str:
    path = tmp_path / name
    (pool_frame() if frame is None else frame).to_pickle(path, compression="gzip")
    return str(path)


def run_select(tmp_path: Path, *argv: str) -> pd.DataFrame:
    """Run ``select_main`` writing ``selected.pkl.gz`` in ``tmp_path`` and return the stored selection."""
    output = tmp_path / "selected.pkl.gz"
    assert select_main([*argv, "--output", str(output)]) == 0
    return pd.read_pickle(output, compression="gzip")


# --- physical values: the gamma window --------------------------------------------------------------------------------


def window_structures(gamma_min: float, gamma_max: float | None = None) -> set[str]:
    """The names of the structures with at least one atom with ``gamma_min <= gamma <= gamma_max`` (hand count)."""
    names = set()
    for _, row in pool_frame().iterrows():
        gamma = row["gamma"]
        inside = (gamma >= gamma_min) & (
            (gamma <= gamma_max) if gamma_max is not None else True
        )
        if inside.any():
            names.add(row["name"])
    return names


@pytest.mark.parametrize("gamma_min", [1.0, 2.5, 3.5, 4.25])
def test_random_extrap_selects_exactly_the_structures_with_an_atom_in_the_window(
    tmp_path, artifact, gamma_min
):
    window = window_structures(gamma_min)
    selected = run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path),
        "--uq",
        artifact,
        "-n",
        str(len(window)),
        "--no-element-stratified",
        "--gamma-min",
        str(gamma_min),
    )
    assert set(selected["name"]) == window
    assert len(selected) == len(window)


def test_gamma_max_closes_the_window_from_above(tmp_path, artifact):
    # atom 3 has gamma 2 + s/4 (in the window for s = 0..3), atom 2 has 1 + s/4 (in the window for s = 4..7)
    window = window_structures(2.0, 2.75)
    assert window == {f"s{s}" for s in range(8)}
    selected = run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path),
        "--uq",
        artifact,
        "-n",
        "8",
        "--no-element-stratified",
        "--gamma-min",
        "2.0",
        "--gamma-max",
        "2.75",
    )
    assert set(selected["name"]) == window


def test_when_the_window_is_too_small_the_selection_is_topped_up_by_the_largest_gamma(
    tmp_path, artifact, caplog
):
    with caplog.at_level(logging.WARNING):
        selected = run_select(
            tmp_path,
            "--predicted",
            write_pool(tmp_path),
            "--uq",
            artifact,
            "-n",
            "7",
            "--no-element-stratified",
            "--gamma-min",
            "2.5",
        )
    # s_k has max gamma 2 + k/4: the window (>= 2.5) holds s2..s9; ask for 7 of those 8
    assert len(selected) == 7
    assert set(selected["name"]) <= {f"s{k}" for k in range(2, 10)}
    # top-up case: only s5..s9 in the window with gamma_min 3.25 (2 + k/4 >= 3.25 iff k >= 5)
    topped = run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path),
        "--uq",
        artifact,
        "-n",
        "7",
        "--no-element-stratified",
        "--gamma-min",
        "3.25",
    )
    assert {f"s{k}" for k in range(5, 10)} <= set(topped["name"])
    assert set(topped["name"]) == {
        f"s{k}" for k in range(3, 10)
    }  # s4, s3: the next largest max-gamma
    assert "topping up from unselected pool by gamma_max" in caplog.text


def test_asking_for_more_structures_than_the_pool_returns_the_whole_pool_with_a_warning(
    tmp_path, artifact, caplog
):
    with caplog.at_level(logging.WARNING):
        selected = run_select(
            tmp_path,
            "--predicted",
            write_pool(tmp_path),
            "--uq",
            artifact,
            "-n",
            "25",
            "--no-element-stratified",
        )
    assert sorted(selected["name"]) == sorted(f"s{s}" for s in range(N_STRUCTURES))
    assert "--n-structures=25 exceeds candidate pool size (10)" in caplog.text


def test_asking_for_exactly_the_pool_size_is_not_a_warning(tmp_path, artifact, caplog):
    with caplog.at_level(logging.WARNING):
        selected = run_select(
            tmp_path,
            "--predicted",
            write_pool(tmp_path),
            "--uq",
            artifact,
            "-n",
            str(N_STRUCTURES),
            "--no-element-stratified",
        )
    assert len(selected) == N_STRUCTURES
    assert "exceeds candidate pool size" not in caplog.text


@pytest.mark.parametrize(
    "strategy", ["random-all", "random-extrap", "fps-all", "fps-extrap"]
)
def test_every_strategy_returns_n_distinct_structures_of_the_pool(
    tmp_path, artifact, strategy
):
    selected = run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path),
        "--uq",
        artifact,
        "-n",
        "4",
        "--no-element-stratified",
        "--strategy",
        strategy,
        "--gamma-min",
        "0.5",
    )
    assert len(selected) == 4
    assert selected["name"].is_unique
    assert set(selected["name"]) <= {f"s{s}" for s in range(N_STRUCTURES)}


def test_the_seed_fixes_the_random_selection(tmp_path, artifact):
    pool = write_pool(tmp_path)
    first = run_select(
        tmp_path,
        "--predicted",
        pool,
        "--uq",
        artifact,
        "-n",
        "4",
        "--no-element-stratified",
        "--strategy",
        "random-all",
        "--seed",
        "3",
    )
    again = run_select(
        tmp_path,
        "--predicted",
        pool,
        "--uq",
        artifact,
        "-n",
        "4",
        "--no-element-stratified",
        "--strategy",
        "random-all",
        "--seed",
        "3",
    )
    other = run_select(
        tmp_path,
        "--predicted",
        pool,
        "--uq",
        artifact,
        "-n",
        "4",
        "--no-element-stratified",
        "--strategy",
        "random-all",
        "--seed",
        "4",
    )
    assert list(first["name"]) == list(again["name"])
    assert list(first["name"]) != list(other["name"])


def test_fps_extrap_picks_only_structures_with_an_atom_in_the_window(
    tmp_path, artifact
):
    window = window_structures(3.0)
    selected = run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path),
        "--uq",
        artifact,
        "-n",
        str(len(window)),
        "--no-element-stratified",
        "--strategy",
        "fps-extrap",
        "--gamma-min",
        "3.0",
    )
    assert set(selected["name"]) == window


# --- physical values: the stratification floor ---------------------------------------------------------------------------


def two_species_pool() -> pd.DataFrame:
    """Six Mo-only structures of six atoms with high gamma and four Nb-only structures of four atoms with low gamma."""
    rows = []
    for s in range(10):
        symbol, n_atoms = ("Mo", 6) if s < 6 else ("Nb", 4)
        gamma = np.full(n_atoms, 5.0 + s if symbol == "Mo" else 0.1)
        rows.append({
            "ase_atoms": Atoms(
                [symbol] * n_atoms, positions=[[1.5 * i, 0, 0] for i in range(n_atoms)]
            ),
            "name": f"{symbol}{s}",
            "gamma": gamma,
        })
    return pd.DataFrame(rows)


def count_atoms(selected: pd.DataFrame, symbol: str) -> int:
    return sum(a.get_chemical_symbols().count(symbol) for a in selected["ase_atoms"])


def test_without_stratification_only_the_extrapolating_structures_are_chosen(
    tmp_path, artifact
):
    selected = run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path, two_species_pool()),
        "--uq",
        artifact,
        "-n",
        "3",
        "--no-element-stratified",
    )
    assert count_atoms(selected, "Nb") == 0


def test_the_stratification_floor_is_met_by_swapping_in_structures_rich_in_the_missing_element(
    tmp_path, artifact
):
    selected = run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path, two_species_pool()),
        "--uq",
        artifact,
        "-n",
        "3",
        "--min-per-element",
        "4",
    )
    assert len(selected) == 3
    assert count_atoms(selected, "Nb") >= 4
    assert count_atoms(selected, "Mo") >= 4


def test_the_default_floor_is_d_plus_one_from_the_artifact(tmp_path, artifact):
    # Nb comes in structures of 4 atoms: a floor of D + 1 = 5 needs two of them (8 atoms), a floor of D = 4 only one
    selected = run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path, two_species_pool()),
        "--uq",
        artifact,
        "-n",
        "3",
    )
    assert count_atoms(selected, "Nb") >= FEATURE_DIM + 1
    assert count_atoms(selected, "Mo") >= FEATURE_DIM + 1


def test_a_floor_larger_than_the_supply_is_clamped_with_a_warning(
    tmp_path, artifact, caplog
):
    frame = (
        two_species_pool().iloc[[0, 1, 6]].reset_index(drop=True)
    )  # 12 Mo atoms, 6 Nb atoms
    with caplog.at_level(logging.WARNING):
        selected = run_select(
            tmp_path,
            "--predicted",
            write_pool(tmp_path, frame),
            "--uq",
            artifact,
            "-n",
            "3",
            "--min-per-element",
            "50",
        )
    assert len(selected) == 3
    assert "Clamped floor for 2 element(s) below 50" in caplog.text


# --- where the element map and D come from ---------------------------------------------------------------------------------


def test_artifact_is_reused_as_the_uq_source_when_uq_is_not_given(
    tmp_path, artifact, caplog
):
    with caplog.at_level(logging.INFO):
        selected = run_select(
            tmp_path,
            "--predicted",
            write_pool(tmp_path),
            "--artifact",
            artifact,
            "-n",
            "3",
            "--no-element-stratified",
        )
    assert len(selected) == 3
    assert (
        f"--uq not given; reusing --artifact ({artifact}) as the UQ source."
        in caplog.text
    )


def test_a_saved_model_gives_the_element_map_and_the_features_give_d(tmp_path, caplog):
    frame = pool_frame()
    frame["ase_atoms"] = [
        Atoms(
            ["Mo", "Nb", "Ta", "W"],
            positions=[[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]],
        )
        for _ in range(len(frame))
    ]
    with caplog.at_level(logging.INFO):
        selected = run_select(
            tmp_path,
            "--predicted",
            write_pool(tmp_path, frame),
            "--model",
            str(SAVED_MODEL),
            "-n",
            "6",
            "--min-per-element",
            "2",
        )
    assert len(selected) == 6
    assert "Derived element_map from" in caplog.text
    assert f"inferred D={FEATURE_DIM} from `features`" in caplog.text


def test_a_saved_model_without_features_needs_an_explicit_floor_or_no_stratification(
    tmp_path,
):
    frame = pool_frame(with_features=False)
    pool = write_pool(tmp_path, frame)
    with pytest.raises(SystemExit, match="Stratification is ON but D is unknown"):
        select_main([
            "--predicted",
            pool,
            "--model",
            str(SAVED_MODEL),
            "-n",
            "3",
            "--output",
            str(tmp_path / "o.pkl.gz"),
        ])
    selected = run_select(
        tmp_path,
        "--predicted",
        pool,
        "--model",
        str(SAVED_MODEL),
        "-n",
        "3",
        "--no-element-stratified",
    )
    assert len(selected) == 3


def test_a_model_directory_without_metadata_cannot_give_the_element_map(tmp_path):
    model = tmp_path / "model_dir"
    model.mkdir()
    with pytest.raises(SystemExit, match="could not derive element_map from --model"):
        select_main([
            "--predicted",
            write_pool(tmp_path),
            "--model",
            str(model),
            "-n",
            "3",
            "--output",
            str(tmp_path / "o.pkl.gz"),
        ])


def test_atoms_of_elements_outside_the_map_are_ignored_by_the_floor_with_a_warning(
    tmp_path, artifact, caplog
):
    frame = pool_frame()
    frame.at[0, "ase_atoms"] = Atoms(
        ["Mo", "Mo", "Nb", "Cu"],
        positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]],
    )
    with caplog.at_level(logging.WARNING):
        run_select(
            tmp_path,
            "--predicted",
            write_pool(tmp_path, frame),
            "--uq",
            artifact,
            "-n",
            "3",
            "--min-per-element",
            "2",
        )
    assert (
        "1 candidate atoms have an element not in the model's element_map"
        in caplog.text
    )


# --- output ------------------------------------------------------------------------------------------------------------------


def test_the_saved_selection_drops_the_prediction_columns_and_keeps_the_others(
    tmp_path, artifact
):
    selected = run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path),
        "--uq",
        artifact,
        "-n",
        "3",
        "--no-element-stratified",
    )
    assert list(selected.columns) == ["ase_atoms", "name"]


def test_an_extxyz_output_holds_the_selected_structures(tmp_path, artifact):
    output = tmp_path / "selected.xyz"
    assert (
        select_main([
            "--predicted",
            write_pool(tmp_path),
            "--uq",
            artifact,
            "-n",
            "3",
            "--no-element-stratified",
            "--output",
            str(output),
        ])
        == 0
    )
    assert len(read(output, index=":")) == 3


def test_relative_paths_are_resolved_against_the_working_directory(
    tmp_path, artifact, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    write_pool(tmp_path)
    assert (
        select_main([
            "--predicted",
            "predicted.pkl.gz",
            "--uq",
            Path(artifact).name,
            "-n",
            "2",
            "--no-element-stratified",
        ])
        == 0
    )
    assert (tmp_path / "selected.pkl.gz").is_file()


def test_the_selection_summary_is_printed(tmp_path, artifact, capsys):
    run_select(
        tmp_path,
        "--predicted",
        write_pool(tmp_path),
        "--uq",
        artifact,
        "-n",
        "3",
        "--no-element-stratified",
    )
    assert capsys.readouterr().out.strip() != ""


# --- errors ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["-n", "3"], "either --predicted or --dataset must be given."),
        (
            ["-n", "3", "--dataset", "d.pkl.gz"],
            "--model is required when running prediction inside select",
        ),
    ],
)
def test_the_source_of_the_candidates_is_required(tmp_path, argv, message):
    with pytest.raises(SystemExit, match=message):
        select_main([
            *argv,
            "--no-element-stratified",
            "--output",
            str(tmp_path / "o.pkl.gz"),
        ])


def test_a_pool_without_ase_atoms_or_gamma_is_rejected(tmp_path, artifact):
    for missing, message in (
        ("ase_atoms", "requires `ase_atoms` column"),
        ("gamma", "requires `gamma` column"),
    ):
        frame = pool_frame().drop(columns=missing)
        with pytest.raises(SystemExit, match=message):
            select_main([
                "--predicted",
                write_pool(tmp_path, frame),
                "--uq",
                artifact,
                "-n",
                "2",
                "--no-element-stratified",
                "--output",
                str(tmp_path / "o.pkl.gz"),
            ])


def test_a_gamma_array_of_the_wrong_length_is_rejected(tmp_path, artifact):
    frame = pool_frame()
    frame.at[2, "gamma"] = np.zeros(3)
    with pytest.raises(SystemExit, match="row 2: len\\(gamma\\)=3 but n_atoms=4"):
        select_main([
            "--predicted",
            write_pool(tmp_path, frame),
            "--uq",
            artifact,
            "-n",
            "2",
            "--no-element-stratified",
            "--output",
            str(tmp_path / "o.pkl.gz"),
        ])


def test_fps_needs_features_of_the_right_shape(tmp_path, artifact):
    without = pool_frame(with_features=False)
    with pytest.raises(SystemExit, match="needs per-atom `features`"):
        select_main([
            "--predicted",
            write_pool(tmp_path, without),
            "--uq",
            artifact,
            "-n",
            "2",
            "--no-element-stratified",
            "--strategy",
            "fps-all",
            "--output",
            str(tmp_path / "o.pkl.gz"),
        ])
    wrong = pool_frame()
    wrong.at[1, "features"] = np.zeros((3, FEATURE_DIM), dtype=np.float32)
    with pytest.raises(
        SystemExit, match=r"row 1: features shape \(3, 4\) but n_atoms=4"
    ):
        select_main([
            "--predicted",
            write_pool(tmp_path, wrong),
            "--uq",
            artifact,
            "-n",
            "2",
            "--no-element-stratified",
            "--strategy",
            "fps-all",
            "--output",
            str(tmp_path / "o.pkl.gz"),
        ])


def test_a_pool_of_structures_without_atoms_is_rejected(tmp_path, artifact):
    frame = pd.DataFrame({
        "ase_atoms": [Atoms(), Atoms()],
        "gamma": [np.zeros(0), np.zeros(0)],
    })
    with pytest.raises(SystemExit, match="candidate atom pool is empty"):
        select_main([
            "--predicted",
            write_pool(tmp_path, frame),
            "--uq",
            artifact,
            "-n",
            "2",
            "--no-element-stratified",
            "--output",
            str(tmp_path / "o.pkl.gz"),
        ])


def test_a_pool_without_any_row_fails_before_the_friendly_message_can_be_given(
    tmp_path, artifact
):
    # Current behaviour, reported on the CLI1 issue: np.concatenate of an empty list raises before the emptiness check.
    frame = pool_frame().iloc[:0]
    with pytest.raises(ValueError, match="need at least one array to concatenate"):
        select_main([
            "--predicted",
            write_pool(tmp_path, frame),
            "--uq",
            artifact,
            "-n",
            "2",
            "--no-element-stratified",
            "--output",
            str(tmp_path / "o.pkl.gz"),
        ])


def test_the_number_of_structures_is_required(capsys):
    with pytest.raises(SystemExit) as stop:
        select_main([])
    assert stop.value.code == 2
    assert (
        "the following arguments are required: -n/--n-structures"
        in capsys.readouterr().err
    )


def test_an_unknown_strategy_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as stop:
        select_main(["-n", "2", "--strategy", "magic"])
    assert stop.value.code == 2
    assert "invalid choice: 'magic'" in capsys.readouterr().err


def test_the_default_strategy_is_random_extrap_with_stratification_on():
    args = select_mod._build_parser().parse_args(["-n", "2"])
    assert (
        args.strategy,
        args.element_stratified,
        args.gamma_min,
        args.gamma_max,
        args.seed,
    ) == ("random-extrap", True, 1.0, None, 42)
