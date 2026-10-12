"""Tests of ``tensorpotential/uq/feature_extraction.py``: the per-atom weight builder, the progress printer,
the element map, ``extract_features`` without an element map, the batched iterator with weights and a progress
report, and the iterator over saved TF-dataset shards.

Logic layer: the shapes, defaults, padding, the progress text (the clock is replaced by scripted values, the one
boundary mocked), the ``frac`` sampling of shards (0 keeps nothing, 1 keeps everything) and the ``try/finally``
that closes the progress report. Physical-value layer: features from the batched and the shard iterators equal
the features the model gives for the same batch computed directly, the weights of a batch add up to the weights
of its structures times their atom counts whatever the bucketing order, and the element indices are those of the
atoms. The model comes from the ``uq_setup`` fixture (``tests/conftest.py``), random weights only.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest

from tensorpotential import constants
from tensorpotential.uq import constants as uqc
from tensorpotential.uq import feature_extraction as fe

import tensorflow as tf  # after tensorpotential, which applies the TensorFlow options first

from tests.shared_models import weights_fingerprint
from tests.tolerances import FLOAT64_ARITHMETIC as ARITH

from ase import Atoms


# --- PerAtomWeightBuilder ---


def _atoms(symbols="CuCu", weight=None) -> Atoms:
    atoms = Atoms(
        symbols, positions=[[0, 0, 0], [0, 0, 2.5], [2.5, 0, 0]][: len(symbols) // 2]
    )
    if weight is not None:
        atoms.info[uqc.UQ_WEIGHT] = weight
    return atoms


def test_the_builder_gives_every_atom_the_weight_of_its_structure_or_one():
    builder = fe.PerAtomWeightBuilder()
    plain = builder.extract_from_ase_atoms(_atoms("CuCuCu"))
    weighted = builder.extract_from_ase_atoms(_atoms("CuCuCu", weight=2.5))
    np.testing.assert_array_equal(plain[uqc.UQ_WEIGHT], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(weighted[uqc.UQ_WEIGHT], [2.5, 2.5, 2.5])
    assert weighted[uqc.UQ_WEIGHT].dtype == np.float64


def test_a_weight_given_as_an_integer_or_a_string_becomes_a_float():
    builder = fe.PerAtomWeightBuilder()
    assert (
        builder.extract_from_ase_atoms(_atoms("CuCu", weight=3))[uqc.UQ_WEIGHT].dtype
        == np.float64
    )
    np.testing.assert_array_equal(
        builder.extract_from_ase_atoms(_atoms("CuCu", weight="4"))[uqc.UQ_WEIGHT],
        [4.0, 4.0],
    )


def test_a_row_is_read_through_its_atoms_object():
    row = {"ase_atoms": _atoms("CuCu", weight=0.5)}
    np.testing.assert_array_equal(
        fe.PerAtomWeightBuilder().extract_from_row(row)[uqc.UQ_WEIGHT], [0.5, 0.5]
    )


def test_the_weights_of_several_structures_are_joined_in_order():
    builder = fe.PerAtomWeightBuilder()
    parts = [{uqc.UQ_WEIGHT: np.array([1.0, 1.0])}, {uqc.UQ_WEIGHT: np.array([3.0])}]
    np.testing.assert_array_equal(
        builder.join_to_batch(parts)[uqc.UQ_WEIGHT], [1.0, 1.0, 3.0]
    )


@pytest.mark.parametrize(
    ("max_atoms", "expected"), [(5, [1, 2, 3, 0, 0]), (3, [1, 2, 3]), (2, [1, 2, 3])]
)
def test_the_weights_are_padded_with_zeros_up_to_the_atom_count_and_never_cut(
    max_atoms, expected
):
    batch = {uqc.UQ_WEIGHT: np.array([1.0, 2.0, 3.0])}
    fe.PerAtomWeightBuilder().pad_batch(batch, {constants.PAD_MAX_N_ATOMS: max_atoms})
    np.testing.assert_array_equal(batch[uqc.UQ_WEIGHT], expected)


# --- ProgressPrinter ---


class _Clock:
    """Scripted ``time.time()``: stands in for the real clock, which cannot be asked for fixed values."""

    def __init__(self, *values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0s"),
        (59.9, "59s"),
        (60, "1m00s"),
        (61, "1m01s"),
        (3599, "59m59s"),
        (3600, "1h00m00s"),
        (3661, "1h01m01s"),
    ],
)
def test_a_duration_is_written_as_hours_minutes_and_seconds_without_empty_leading_units(
    seconds, text
):
    assert fe.ProgressPrinter._fmt(seconds) == text


def test_a_progress_line_shows_percent_count_elapsed_time_and_the_remaining_time(
    monkeypatch, capsys
):
    monkeypatch.setattr(fe.time, "time", _Clock(100.0, 110.0))
    printer = fe.ProgressPrinter(total=100, desc="Work", unit="structs", interval=0.0)
    printer.update(25)  # clock 110: 10 s elapsed, a quarter done: 30 s to go
    assert (
        capsys.readouterr().out
        == "Work | 25% (25/100 structs) | elapsed 10s | ETA ~30s\n"
    )


def test_a_progress_line_without_a_total_shows_only_the_count(monkeypatch, capsys):
    monkeypatch.setattr(fe.time, "time", _Clock(0.0, 61.0))
    printer = fe.ProgressPrinter(desc="Count", unit="batches", interval=0.0)
    printer.update(4)
    assert capsys.readouterr().out == "Count | 4 batches | elapsed 1m01s\n"


def test_a_total_of_zero_is_treated_as_no_total(monkeypatch, capsys):
    monkeypatch.setattr(fe.time, "time", _Clock(0.0, 5.0))
    fe.ProgressPrinter(total=0, desc="Z", interval=0.0).update(2)
    assert capsys.readouterr().out == "Z | 2 items | elapsed 5s\n"


def test_updates_inside_the_interval_print_nothing_and_the_next_one_after_it_prints(
    monkeypatch, capsys
):
    monkeypatch.setattr(fe.time, "time", _Clock(0.0, 3.0, 12.0, 14.0))
    printer = fe.ProgressPrinter(total=10, desc="P", unit="u", interval=10.0)
    printer.update(
        1
    )  # clock 3: the first line comes only after the interval from the start of time
    first = capsys.readouterr().out
    printer.update(1)  # clock 12
    second = capsys.readouterr().out
    printer.update(1)  # clock 14: 2 s after the last line
    third = capsys.readouterr().out
    assert first == "" and third == ""
    assert second == "P | 20% (2/10 u) | elapsed 12s | ETA ~48s\n"


def test_closing_prints_the_final_line_with_full_percent_and_done(monkeypatch, capsys):
    monkeypatch.setattr(fe.time, "time", _Clock(0.0, 90.0))
    printer = fe.ProgressPrinter(total=40, desc="Fin", unit="u", interval=1e9)
    printer.update(10)  # no line: inside the interval
    capsys.readouterr()
    printer.close()
    assert capsys.readouterr().out == "Fin | 100% (10/40 u) | elapsed 1m30s | done\n"


def test_closing_without_a_total_says_done_after_the_count(monkeypatch, capsys):
    monkeypatch.setattr(fe.time, "time", _Clock(0.0, 7.0))
    printer = fe.ProgressPrinter(desc="N", unit="u")
    printer.close()
    assert capsys.readouterr().out == "N | 0 u | elapsed 7s | done\n"


def test_the_percentage_is_capped_at_one_hundred(monkeypatch, capsys):
    monkeypatch.setattr(fe.time, "time", _Clock(0.0, 1.0))
    fe.ProgressPrinter(total=4, desc="C", unit="u", interval=0.0).update(9)
    assert "100% (9/4 u)" in capsys.readouterr().out


# --- the element map ---


def test_the_element_map_is_alphabetical_over_all_structures():
    atoms = [Atoms("ZnCu"), Atoms("Al"), Atoms("CuCu")]
    assert fe._build_element_map(atoms) == {"Al": 0, "Cu": 1, "Zn": 2}
    assert fe._build_element_map(iter(atoms)) == {"Al": 0, "Cu": 1, "Zn": 2}
    assert fe._build_element_map([]) == {}


# --- the model ---


@pytest.fixture(scope="module")
def feature_model(uq_setup):
    """The UQ model with the feature instruction, compute function decorated; read-only for the tests."""
    from tensorpotential.instructions import load_instructions
    from tensorpotential.tensorpot import TensorPotential
    from tensorpotential.uq.factories import patch_instructions_for_basis_rp_features

    instructions = load_instructions(uq_setup["model_yaml"])
    patch_instructions_for_basis_rp_features(
        instructions, out_dim=uq_setup["feature_spec"]["out_dim"]
    )
    tp = TensorPotential(instructions, param_dtype=tf.float64)
    tp.load_checkpoint(uq_setup["checkpoint"])
    tp.model.decorate_compute_function()
    tp.model.compute_function.extra_return_keys = [uqc.FEATURES]
    before = weights_fingerprint(tp.model)
    yield tp.model
    assert weights_fingerprint(tp.model) == before, "a test changed the shared model"


@pytest.fixture(scope="module")
def calculator(uq_setup):
    return fe.setup_feature_calculator(
        uq_setup["model_yaml"],
        uq_setup["checkpoint"],
        param_dtype=tf.float64,
        feature_spec=uq_setup["feature_spec"],
    )


def _structures(uq_setup, n=5, weights=None):
    atoms = [a.copy() for a in uq_setup["atoms"][:n]]
    for i, a in enumerate(atoms):
        if weights is not None:
            a.info[uqc.UQ_WEIGHT] = weights[i]
    return atoms


# --- extract_features ---


def test_without_an_element_map_the_features_use_the_alphabetical_one_of_all_structures(
    uq_setup, calculator
):
    atoms = _structures(uq_setup, n=3)
    from_generator = list(
        fe.extract_features(calculator, (a for a in atoms))
    )  # a generator is read twice
    explicit = list(
        fe.extract_features(
            calculator,
            [a.copy() for a in atoms],
            element_map=fe._build_element_map(atoms),
        )
    )
    assert len(from_generator) == 3
    for (feat_a, idx_a), (feat_b, idx_b), at in zip(from_generator, explicit, atoms):
        assert feat_a.shape == (len(at), uq_setup["feature_dim"])
        np.testing.assert_allclose(feat_a, feat_b, rtol=ARITH.rtol, atol=ARITH.atol)
        np.testing.assert_array_equal(idx_a, idx_b)
        symbols = sorted(set(sum((a.get_chemical_symbols() for a in atoms), [])))
        np.testing.assert_array_equal(
            idx_a, [symbols.index(s) for s in at.get_chemical_symbols()]
        )


def test_the_features_of_a_structure_are_attached_to_a_calculator_on_the_structure(
    uq_setup, calculator
):
    atoms = _structures(uq_setup, n=1)
    next(fe.extract_features(calculator, atoms, element_map=uq_setup["element_map"]))
    assert atoms[0].calc is calculator


# --- the batched iterator ---


def _batched(uq_setup, model, atoms, **kwargs):
    return list(
        fe.batched_feature_iterator(
            atoms,
            model,
            element_map=uq_setup["element_map"],
            cutoff=6.0,
            verbose=False,
            **kwargs,
        )
    )


def test_the_weights_of_the_batches_follow_the_structures_whatever_the_bucketing_order(
    uq_setup, feature_model
):
    weights = [1.0, 2.0, 0.5, 4.0, 3.0]
    atoms = _structures(uq_setup, weights=weights)
    results = _batched(uq_setup, feature_model, atoms)

    expected_total = sum(w * len(a) for w, a in zip(weights, atoms))
    assert sum(w.sum() for _, _, w in results) == pytest.approx(
        expected_total, rel=ARITH.rtol
    )
    # per weight value, the number of atoms carrying it is the atom count of the structure that has it
    found = np.concatenate([w for _, _, w in results])
    for weight, atom in zip(weights, atoms):
        assert np.sum(found == weight) == len(atom)
    assert all(w.dtype == np.float64 for _, _, w in results)


def test_the_element_indices_of_the_batches_are_those_of_the_atoms(
    uq_setup, feature_model
):
    atoms = _structures(uq_setup)
    results = _batched(uq_setup, feature_model, atoms)
    element_map = uq_setup["element_map"]
    expected = np.bincount(
        [element_map[s] for a in atoms for s in a.get_chemical_symbols()], minlength=4
    )
    found = np.bincount(np.concatenate([idx for _, idx, _ in results]), minlength=4)
    np.testing.assert_array_equal(found, expected)


def test_the_batched_features_equal_the_per_structure_features_of_the_calculator(
    uq_setup, feature_model, calculator
):
    atoms = _structures(uq_setup, n=3)
    batched = np.concatenate([
        f for f, _, _ in _batched(uq_setup, feature_model, atoms)
    ])
    single = np.concatenate([
        f
        for f, _ in fe.extract_features(
            calculator, [a.copy() for a in atoms], element_map=uq_setup["element_map"]
        )
    ])
    # the same atoms in another order: compare the sorted rows through their norms and sums
    np.testing.assert_allclose(
        np.sort(np.linalg.norm(batched, axis=1)),
        np.sort(np.linalg.norm(single, axis=1)),
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.sort(batched.sum(axis=1)), np.sort(single.sum(axis=1)), rtol=1e-8, atol=1e-10
    )


class _CountedAtoms:
    """An iterable of structures without ``len``, as a streaming source is."""

    def __init__(self, atoms):
        self._atoms = atoms

    def __iter__(self):
        return iter(self._atoms)


class _Estimate:
    """What a streaming source reports for its size: an object with a ``total`` that refines."""

    def __init__(self, total):
        self.total = total


def test_a_source_without_length_takes_its_size_from_the_argument_and_reports_progress(
    uq_setup, feature_model, capsys
):
    atoms = _structures(uq_setup, n=3)
    results = list(
        fe.batched_feature_iterator(
            _CountedAtoms(atoms),
            feature_model,
            element_map=uq_setup["element_map"],
            cutoff=6.0,
            verbose=True,
            desc="Feat",
            total_num_structures=3,
        )
    )
    out = capsys.readouterr().out.strip().splitlines()
    assert sum(len(f) for f, _, _ in results) == sum(len(a) for a in atoms)
    assert out[-1].startswith("Feat | 100% (3/3 structs)") and out[-1].endswith(
        "| done"
    )


def test_a_size_estimate_that_refines_is_followed_by_the_progress_report(
    uq_setup, feature_model, capsys
):
    atoms = _structures(uq_setup, n=2)
    estimate = _Estimate(total=2)
    list(
        fe.batched_feature_iterator(
            _CountedAtoms(atoms),
            feature_model,
            element_map=uq_setup["element_map"],
            cutoff=6.0,
            verbose=True,
            total_num_structures=estimate,
        )
    )
    assert "(2/2 structs)" in capsys.readouterr().out.splitlines()[-1]


def test_the_progress_report_is_closed_when_the_consumer_stops_early(
    uq_setup, feature_model, capsys
):
    atoms = _structures(uq_setup, n=5)
    iterator = fe.batched_feature_iterator(
        atoms,
        feature_model,
        element_map=uq_setup["element_map"],
        cutoff=6.0,
        verbose=True,
        max_num_neighbours_per_batch=200,
    )
    next(iterator)
    iterator.close()
    assert capsys.readouterr().out.strip().endswith("done")


def test_a_quiet_run_prints_nothing(uq_setup, feature_model, capsys):
    _batched(uq_setup, feature_model, _structures(uq_setup, n=2))
    assert capsys.readouterr().out == ""


# --- the shard iterator ---


def _save_shard(path, batches):
    """Save batches (dicts of numpy arrays) as a GZIP TF dataset, one element per batch."""
    dataset = None
    for batch in batches:
        element = tf.data.Dataset.from_tensors({
            k: tf.constant(v) for k, v in batch.items()
        })
        dataset = element if dataset is None else dataset.concatenate(element)
    tf.data.Dataset.save(dataset, str(path), compression="GZIP")
    return str(path)


@pytest.fixture(scope="module")
def shard_batches(uq_setup):
    """Two padded batches of structures, as ``grace_preprocess`` shards hold them."""
    from tensorpotential.data.databuilder import (
        GeometricalDataBuilder,
        construct_batches,
    )

    builder = GeometricalDataBuilder(uq_setup["element_map"], cutoff=6.0)
    atoms = _structures(uq_setup, n=4)
    batches = construct_batches(atoms, [builder], batch_size=2, verbose=False)
    return list(batches)


def _direct_features(model, batch):
    keys = {
        k: fe.COMPUTE_SPEC_DTYPE_MAP[v["dtype"]] for k, v in model.compute_specs.items()
    }
    out = model.compute({
        k: tf.constant(batch[k], dtype=dt) for k, dt in keys.items() if k in batch
    })
    n_real = int(batch[constants.N_ATOMS_BATCH_REAL])
    return out[uqc.FEATURES][:n_real].numpy(), np.asarray(batch[constants.ATOMIC_MU_I])[
        :n_real
    ]


def test_the_shard_iterator_gives_the_features_of_every_batch_with_unit_weights(
    tmp_path, feature_model, shard_batches
):
    shard = _save_shard(tmp_path / "shard", shard_batches)
    results = list(
        fe.tf_dataset_feature_iterator([shard], feature_model, verbose=False)
    )
    assert len(results) == len(shard_batches)
    for (features, elements, weights), batch in zip(results, shard_batches):
        direct, direct_elements = _direct_features(feature_model, batch)
        np.testing.assert_allclose(features, direct, rtol=ARITH.rtol, atol=ARITH.atol)
        np.testing.assert_array_equal(elements, direct_elements)
        np.testing.assert_array_equal(weights, np.ones(len(features)))
        assert weights.dtype == np.float64


def test_shards_are_read_in_the_order_given(tmp_path, feature_model, shard_batches):
    first = _save_shard(tmp_path / "a", shard_batches[:1])
    second = _save_shard(tmp_path / "b", shard_batches[1:])
    forward = list(
        fe.tf_dataset_feature_iterator([first, second], feature_model, verbose=False)
    )
    backward = list(
        fe.tf_dataset_feature_iterator([second, first], feature_model, verbose=False)
    )
    assert [len(f) for f, _, _ in forward] == [len(f) for f, _, _ in reversed(backward)]
    np.testing.assert_array_equal(forward[0][0], backward[1][0])


def test_no_shards_give_no_batches_and_no_progress_report(feature_model, capsys):
    assert list(fe.tf_dataset_feature_iterator([], feature_model, verbose=True)) == []
    assert capsys.readouterr().out == ""


def test_a_sampled_fraction_of_zero_skips_every_batch_and_one_keeps_every_batch(
    tmp_path, feature_model, shard_batches
):
    shard = _save_shard(tmp_path / "s", shard_batches)
    assert (
        list(
            fe.tf_dataset_feature_iterator(
                [shard], feature_model, frac=0.0, verbose=False
            )
        )
        == []
    )
    kept = list(
        fe.tf_dataset_feature_iterator([shard], feature_model, frac=1.0, verbose=False)
    )
    assert len(kept) == len(shard_batches)


def test_a_sampled_fraction_is_reproducible_with_the_seed(
    tmp_path, feature_model, shard_batches
):
    shard = _save_shard(tmp_path / "s", shard_batches * 8)  # 16 batches
    runs = [
        [
            f.shape[0]
            for f, _, _ in fe.tf_dataset_feature_iterator(
                [shard], feature_model, frac=0.5, seed=s, verbose=False
            )
        ]
        for s in (1, 1, 2)
    ]
    assert runs[0] == runs[1]
    assert 0 < len(runs[0]) < 16  # a half, not all or none, for a fair draw


def test_the_shard_progress_counts_batches_also_when_they_are_skipped(
    tmp_path, feature_model, shard_batches, capsys
):
    shard = _save_shard(tmp_path / "s", shard_batches)
    list(
        fe.tf_dataset_feature_iterator(
            [shard],
            feature_model,
            frac=0.0,
            verbose=True,
            total_num_batches=2,
            desc="Sh",
        )
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("Sh | 100% (2/2 batches)") and lines[-1].endswith(
        "| done"
    )


def test_the_shard_progress_counts_the_batches_that_are_read(
    tmp_path, feature_model, shard_batches, capsys
):
    shard = _save_shard(tmp_path / "s", shard_batches)
    results = list(
        fe.tf_dataset_feature_iterator(
            [shard], feature_model, verbose=True, total_num_batches=2, desc="Rd"
        )
    )
    assert len(results) == 2
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("Rd | 100% (2/2 batches)") and last.endswith("| done")
