"""The UQ paths that read an artifact into a calculator or bake it into a kokkos file.

Logic layer: ``TPCalculator.load_uq_artifacts`` (a mutable model and a saved model, both compute modes, every error) and
``export_kokkos._merge_uq_artifacts`` (every check of the alignment between artifact and model). Physical layer: the sigma
of a structure read through ``load_uq_artifacts`` equals the sigma of the factory calculator built from the same files, the
energy is the energy of the model without UQ, and the arrays written for the kokkos pair style are the arrays stored in the
artifact (read with ``numpy`` and not through ``GMMUQModel``).
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from tensorpotential.calculator.asecalculator import TPCalculator
from tensorpotential.export_kokkos import _merge_uq_artifacts
from tensorpotential.instructions import load_instructions
from tensorpotential.tensorpot import TensorPotential
from tensorpotential.uq import constants as uqc
from tensorpotential.uq.compute import (
    ComputeStructureEnergyAndForcesAndVirialAndGammaOnly,
    ComputeStructureEnergyAndForcesAndVirialAndUncertainty,
)
from tensorpotential.uq.factories import get_gmm_uq_calculator

N_ELEMENTS = 4


def _edited_artifact(tmp_path, uq_setup, name, edit):
    """Copy the artifact of the fixture with ``edit(raw)`` applied to its arrays."""
    raw = dict(np.load(uq_setup["artifact_path"], allow_pickle=True))
    edit(raw)
    path = str(tmp_path / f"{name}.npz")
    np.savez(path, **raw)
    return path


def _with_thresholds(raw):
    n_clusters = raw[f"{uqc.CENTROIDS}_{int(raw['elements'][0])}"].shape[0]
    raw["interp_thresholds"] = np.full((N_ELEMENTS, n_clusters), 1.0, dtype=np.float64)


@pytest.fixture(scope="module")
def artifact_with_thresholds(tmp_path_factory, uq_setup):
    return _edited_artifact(
        tmp_path_factory.mktemp("merge"), uq_setup, "with_thresholds", _with_thresholds
    )


@pytest.fixture(scope="module")
def instruction_dict(uq_setup):
    return load_instructions(uq_setup["model_yaml"])


def _model(uq_setup, *, decorated=True):
    """The model of the fixture; ``compute`` is traced unless ``decorated`` is false."""
    tp = TensorPotential(
        load_instructions(uq_setup["model_yaml"]), param_dtype=tf.float64
    )
    tp.load_checkpoint(uq_setup["checkpoint"])
    if decorated:
        tp.model.decorate_compute_function()
    return tp.model


def _atoms(uq_setup, calculator):
    atoms = uq_setup["atoms"][0].copy()
    atoms.calc = calculator
    return atoms


@pytest.fixture(scope="module")
def reference(uq_setup):
    """Energy of the model without UQ and the sigma of the factory calculator, for the first structure."""
    plain = _atoms(uq_setup, TPCalculator(model=_model(uq_setup)))
    energy = plain.get_potential_energy()
    factory = get_gmm_uq_calculator(
        model_yaml=uq_setup["model_yaml"],
        checkpoint=uq_setup["checkpoint"],
        gmm_artifact_path=uq_setup["artifact_path"],
        param_dtype=tf.float64,
    )
    uq_atoms = _atoms(uq_setup, factory)
    uq_atoms.get_potential_energy()
    return {
        "energy": energy,
        "atomic_sigma": np.asarray(factory.results[uqc.ATOMIC_SIGMA]),
        "total_sigma": float(factory.results[uqc.TOTAL_SIGMA]),
    }


# ---------------------------------------------------------------- TPCalculator.load_uq_artifacts


def test_a_mutable_model_gets_the_sigma_of_the_factory_calculator_and_keeps_its_energy(
    uq_setup, reference
):
    calc = TPCalculator(model=_model(uq_setup))
    calc.load_uq_artifacts(uq_setup["artifact_path"])
    atoms = _atoms(uq_setup, calc)
    np.testing.assert_allclose(
        atoms.get_potential_energy(), reference["energy"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        calc.results[uqc.ATOMIC_SIGMA], reference["atomic_sigma"], rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        float(calc.results[uqc.TOTAL_SIGMA]), reference["total_sigma"], rtol=1e-8
    )
    assert uqc.DSIGMA_DR in calc.results
    assert isinstance(
        calc.models[0].compute_function,
        ComputeStructureEnergyAndForcesAndVirialAndUncertainty,
    )
    assert calc.gmm_uq_model is not None


def test_a_mutable_model_without_the_sigma_gradient_drops_its_keys(uq_setup, reference):
    calc = TPCalculator(model=_model(uq_setup))
    calc.load_uq_artifacts(uq_setup["artifact_path"], compute_dsigma_dr=False)
    _atoms(uq_setup, calc).get_forces()
    np.testing.assert_allclose(
        calc.results[uqc.ATOMIC_SIGMA], reference["atomic_sigma"], rtol=1e-8, atol=1e-10
    )
    assert uqc.DSIGMA_DR not in calc.results
    assert uqc.VIRIAL_SIGMA not in calc.results
    assert isinstance(
        calc.models[0].compute_function,
        ComputeStructureEnergyAndForcesAndVirialAndGammaOnly,
    )


def test_a_mutable_model_takes_the_dtype_of_the_artifact_loader_when_one_is_given(
    uq_setup, reference
):
    calc = TPCalculator(model=_model(uq_setup))
    calc.load_uq_artifacts(uq_setup["artifact_path"], param_dtype=tf.float64)
    _atoms(uq_setup, calc).get_forces()
    np.testing.assert_allclose(
        calc.results[uqc.ATOMIC_SIGMA], reference["atomic_sigma"], rtol=1e-8, atol=1e-10
    )


def test_a_saved_model_is_rebuilt_from_its_yaml_and_checkpoint(
    uq_setup, reference, tmp_path
):
    export_path = str(tmp_path / "saved")
    _model(uq_setup).save_model(export_path)
    calc = TPCalculator(export_path)
    calc.load_uq_artifacts(
        uq_setup["artifact_path"],
        model_yaml=uq_setup["model_yaml"],
        checkpoint=uq_setup["checkpoint"],
        param_dtype=tf.float64,
    )
    atoms = _atoms(uq_setup, calc)
    np.testing.assert_allclose(
        atoms.get_potential_energy(), reference["energy"], rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        calc.results[uqc.ATOMIC_SIGMA], reference["atomic_sigma"], rtol=1e-8, atol=1e-10
    )


def test_a_saved_model_needs_its_yaml_and_checkpoint(uq_setup, tmp_path):
    export_path = str(tmp_path / "saved")
    _model(uq_setup).save_model(export_path)
    with pytest.raises(ValueError, match="model_yaml and checkpoint are required"):
        TPCalculator(export_path).load_uq_artifacts(uq_setup["artifact_path"])


def test_a_mutable_model_whose_compute_was_never_traced_is_not_supported_yet(uq_setup):
    # Characterization of the present behaviour (finding on the ARCH2 issue): the retrace step deletes the traced
    # ``compute`` of the model, which does not exist before ``decorate_compute_function`` ran.
    calc = TPCalculator(model=_model(uq_setup, decorated=False))
    with pytest.raises(AttributeError, match="compute"):
        calc.load_uq_artifacts(uq_setup["artifact_path"])


@pytest.mark.parametrize("n_models", [0, 2])
def test_anything_but_exactly_one_model_is_refused(n_models, uq_setup):
    model = _model(uq_setup)
    calc = TPCalculator(model=model)
    calc.models = [model] * n_models
    with pytest.raises(ValueError, match="exactly one model"):
        calc.load_uq_artifacts(uq_setup["artifact_path"])


@pytest.mark.parametrize("missing_key", [uqc.UQ_FEATURE_MODE, uqc.UQ_RP_MATRIX])
def test_an_artifact_without_the_basis_rp_feature_or_its_matrix_is_refused_for_a_mutable_model(
    missing_key, uq_setup, tmp_path
):
    path = _edited_artifact(
        tmp_path, uq_setup, "broken", lambda raw: raw.pop(missing_key)
    )
    with pytest.raises(ValueError, match="was not built with the basis-RP feature"):
        TPCalculator(model=_model(uq_setup)).load_uq_artifacts(path)


# ---------------------------------------------------------------- export_kokkos._merge_uq_artifacts


def _weights(uq_setup, **extra):
    return {
        "element_names": np.array(list(uq_setup["element_map"]), dtype="S2"),
        **extra,
    }


def test_the_dense_arrays_written_are_the_arrays_of_the_artifact(
    uq_setup, artifact_with_thresholds, instruction_dict
):
    weights = _weights(uq_setup)
    _merge_uq_artifacts(weights, artifact_with_thresholds, N_ELEMENTS, instruction_dict)
    stored = np.load(artifact_with_thresholds, allow_pickle=True)
    rp_matrix = stored[uqc.UQ_RP_MATRIX]
    assert weights["uq_schema_version"].tolist() == [
        3
    ]  # linear feature: no normalisation, density or transform
    assert weights["uq_n_elements"].tolist() == [N_ELEMENTS]
    assert weights["uq_feature_dim"].tolist() == [rp_matrix.shape[1]]
    assert weights["uq_rp_dim"].tolist() == [int(stored[uqc.UQ_RP_DIM])]
    assert weights["uq_rp_seed"].tolist() == [int(stored[uqc.UQ_RP_SEED])]
    assert weights["uq_rp_matrix"].dtype == np.float32
    np.testing.assert_array_equal(weights["uq_rp_matrix"], rp_matrix.astype(np.float32))
    np.testing.assert_array_equal(
        weights["uq_interp_thresholds"], stored["interp_thresholds"].astype(np.float32)
    )
    assert weights["uq_missing_elem_inv_cov_diag"].tolist() == [1e6]
    assert weights["uq_rp_normalize"].tolist() == [0]
    assert weights["uq_rp_add_density_channel"].tolist() == [0]
    assert weights["uq_feature_transform"].tolist() == [0]
    first_element = int(stored["elements"][0])
    n_clusters = stored[f"{uqc.CENTROIDS}_{first_element}"].shape[0]
    assert weights["uq_n_clusters"][first_element] == n_clusters
    np.testing.assert_array_equal(
        weights["uq_centroids"][first_element, :n_clusters],
        stored[f"{uqc.CENTROIDS}_{first_element}"].astype(np.float32),
    )


def test_an_artifact_for_another_number_of_elements_is_refused(
    uq_setup, artifact_with_thresholds, instruction_dict
):
    with pytest.raises(ValueError, match="element count"):
        _merge_uq_artifacts(
            _weights(uq_setup),
            artifact_with_thresholds,
            N_ELEMENTS - 1,
            instruction_dict,
        )


def test_an_artifact_in_another_element_order_is_refused(
    artifact_with_thresholds, instruction_dict
):
    reordered = {"element_names": np.array([b"Nb", b"Mo", b"Ta", b"W"])}
    with pytest.raises(ValueError, match="element order"):
        _merge_uq_artifacts(
            reordered, artifact_with_thresholds, N_ELEMENTS, instruction_dict
        )


def test_a_model_without_element_names_skips_the_order_check(
    artifact_with_thresholds, instruction_dict
):
    weights: dict = {}
    _merge_uq_artifacts(weights, artifact_with_thresholds, N_ELEMENTS, instruction_dict)
    assert weights["uq_n_elements"].tolist() == [N_ELEMENTS]


def test_an_artifact_without_the_basis_rp_feature_is_refused_for_the_kokkos_export(
    uq_setup, instruction_dict, tmp_path
):
    def edit(raw):
        _with_thresholds(raw)
        raw.pop(uqc.UQ_FEATURE_MODE)

    path = _edited_artifact(tmp_path, uq_setup, "no_rp", edit)
    with pytest.raises(ValueError, match="requires a basis-RP artifact"):
        _merge_uq_artifacts(_weights(uq_setup), path, N_ELEMENTS, instruction_dict)


def test_a_feature_dimension_that_disagrees_with_the_spec_is_refused(
    uq_setup, instruction_dict, tmp_path
):
    def edit(raw):
        _with_thresholds(raw)
        raw[uqc.UQ_RP_DENSITY] = np.array(
            True
        )  # promises density channels the stored features do not have

    path = _edited_artifact(tmp_path, uq_setup, "density", edit)
    with pytest.raises(ValueError, match="feature dim"):
        _merge_uq_artifacts(_weights(uq_setup), path, N_ELEMENTS, instruction_dict)


def test_an_artifact_without_thresholds_is_refused(uq_setup, instruction_dict):
    with pytest.raises(ValueError, match="no interpolation thresholds"):
        _merge_uq_artifacts(
            _weights(uq_setup), uq_setup["artifact_path"], N_ELEMENTS, instruction_dict
        )


def test_an_unknown_feature_transform_is_refused(uq_setup, instruction_dict, tmp_path):
    def edit(raw):
        _with_thresholds(raw)
        raw[uqc.UQ_FEATURE_TRANSFORM] = np.array("cosh")

    path = _edited_artifact(tmp_path, uq_setup, "transform", edit)
    with pytest.raises(ValueError, match="unsupported feature_transform"):
        _merge_uq_artifacts(_weights(uq_setup), path, N_ELEMENTS, instruction_dict)
