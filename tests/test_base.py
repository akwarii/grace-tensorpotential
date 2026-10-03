"""Tests of ``instructions/base.py``: the ``model.yaml`` round trip and the constructor defaults it persists.

``capture_init_args`` writes every constructor argument, defaults included, into the saved ``model.yaml``, and
``load_instructions`` rebuilds the instructions from it, so the saved defaults are part of the API of every model
that exists. Two layers:

- logic: ``str_to_class``, the three yaml formats of ``load_instructions``, ``save_instructions_dict`` and its
  metadata block, the table of constructor defaults of every ``TPInstruction`` subclass;
- physical values: every preset and the yamls of ``tests/`` are built, written, reloaded and rebuilt, and the
  rebuilt model has the same variables and gives the same energy and forces as the original with copied weights.

Quirks of the current code are pinned as they are and say so; none is fixed here.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import inspect
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorflow as tf
import yaml
from ase import Atoms

from tensorpotential import __version__
from tensorpotential.calculator import TPCalculator
from tensorpotential.instructions.base import (
    TPInstruction,
    _build_metadata,
    _extract_instructions_data,
    capture_init_args,
    class_to_str,
    load_instructions,
    save_instructions_dict,
    str_to_class,
)
from tensorpotential.potentials import presets
from tensorpotential.tpmodel import TPModel, extract_cutoff_and_elements
from tests import instruction_signatures as signatures
from tests.seeded_weights import seed_trainable_variables
from tests.tolerances import FLOAT64_ARITHMETIC

TESTS_DIR = Path(__file__).parent

ELEMENT_MAP = {"Cu": 0, "Zn": 1}

# a small element set and small sizes: only the structure of the model matters here, not its accuracy
PRESET_CASES: dict[str, dict[str, Any]] = {
    "LINEAR": {"lmax": 2, "n_rad_max": 8, "embedding_size": 8},
    "FS": {
        "lmax": (2, 2, 2, 2),
        "Lmax": (None, 1, 0, 0),
        "max_sum_l": (None, None, 3, 2),
        "lmax_hist": (None, None, None, 2),
        "n_rad_max": (4, 4, 4, 4),
        "n_rad_base": 6,
        "embedding_size": 8,
    },
    "GRACE_1LAYER_v1_24": {
        "lmax": 2,
        "n_rad_max": 8,
        "embedding_size": 8,
        "n_mlp_dens": 4,
    },
    "GRACE_2LAYER_v1_24": {
        "lmax": (2, 2),
        "n_rad_max": (8, 8),
        "embedding_size": 8,
        "n_mlp_dens": 4,
    },
    "GRACE_2LAYER_scaled": {
        "lmax": (2, 2),
        "n_rad_max": (8, 8),
        "embedding_size": 8,
        "n_mlp_dens": 4,
    },
    "GRACE_1LAYER_v2_25": {
        "lmax": 2,
        "n_rad_max": 8,
        "embedding_size": 8,
        "n_mlp_dens": 4,
        "prod_func_n_max": 8,
    },
    "GRACE_2LAYER_v2_25": {
        "lmax": (2, 2),
        "n_rad_max": (8, 8),
        "embedding_size": 8,
        "n_mlp_dens": 4,
        "prod_func_n_max": (8, 8),
        "indicator_lmax": 1,
    },
}

TEST_YAMLS = [
    "model_grace.yaml",
    "model_grace_2L_omat.yaml",
    "model_grace_2L_omat_large_base.yaml",
]


# ---------------------------------------------------------------------------------------------- helpers


def preset_instructions(name: str) -> dict[str, TPInstruction]:
    """The instructions of one preset on ``ELEMENT_MAP`` (``GRACE_2LAYER_scaled`` is not registered)."""
    return getattr(presets, name)(
        element_map=ELEMENT_MAP, **PRESET_CASES[name]
    ).get_instructions()


def build_model(
    instructions: dict[str, TPInstruction] | list[TPInstruction],
) -> TPModel:
    """A float64 model built from ``instructions`` (each instruction is built once, so pass fresh ones)."""
    model = TPModel(instructions)
    model.build(tf.float64)
    return model


def saved_instructions(path: Path) -> dict[str, dict]:
    """The ``name -> constructor arguments`` block of a model.yaml, whatever its format."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return {d["name"]: d for d in data}
    return {
        k: v for k, v in data.get("instructions", data).items() if k != "__metadata__"
    }


def members(model: TPModel) -> list[TPInstruction]:
    """The instructions of a model in file order (a very old yaml gives a list, the others a dict)."""
    instructions = model.instructions
    return (
        list(instructions.values())
        if isinstance(instructions, dict)
        else list(instructions)
    )


def keyed_variables(model: TPModel) -> dict[tuple[str, int], tf.Variable]:
    """Variables by (name, position among the variables of that name): a name is not always unique."""
    seen: Counter[str] = Counter()
    keyed = {}
    for var in model.variables:
        keyed[var.name, seen[var.name]] = var
        seen[var.name] += 1
    return keyed


def variable_specs(
    model: TPModel,
) -> dict[tuple[str, int], tuple[tuple[int, ...], str, bool]]:
    """Shape, dtype and trainability of every variable, by name."""
    return {
        key: (tuple(v.shape), v.dtype.name, v.trainable)
        for key, v in keyed_variables(model).items()
    }


def cluster(model: TPModel) -> Atoms:
    """Three atoms 2.3 to 2.6 A apart, with the elements of the model in the order of its element map."""
    _, symbols, index = extract_cutoff_and_elements(model.instructions)
    ordered = [str(s) for s in symbols[np.argsort(index)]]
    return Atoms(
        [ordered[i % len(ordered)] for i in range(3)],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.3], [1.9, 0.3, 4.0]],
        pbc=False,
    )


def energy_and_forces(model: TPModel, atoms: Atoms) -> tuple[float, np.ndarray]:
    """Energy (eV) and forces (eV/A) of a copy of ``atoms`` through the ASE calculator."""
    atoms = atoms.copy()
    atoms.calc = TPCalculator(model=model)
    return float(atoms.get_potential_energy()), np.array(atoms.get_forces())


def assert_round_trip(first: TPModel, path: Path, tmp_path: Path) -> None:
    """Write ``first``, reload and rebuild it, and compare structure, arguments, variables and outputs."""
    save_instructions_dict(str(path), first.instructions, param_dtype=tf.float64)
    reloaded = load_instructions(str(path))
    second = build_model(reloaded)

    # the instruction list: order, names and classes
    assert [i.name for i in members(second)] == [i.name for i in members(first)]
    assert [type(i) for i in members(second)] == [type(i) for i in members(first)]

    # the saved constructor arguments: writing the reloaded model gives the same instructions block
    again = tmp_path / "again.yaml"
    save_instructions_dict(str(again), second.instructions, param_dtype=tf.float64)
    assert saved_instructions(again) == saved_instructions(path)

    # variables: names, shapes, dtypes, trainability; the constants (index tables, cutoffs, element maps)
    # are rebuilt from the constructor arguments alone and equal the original ones
    assert variable_specs(second) == variable_specs(first)
    first_variables, second_variables = keyed_variables(first), keyed_variables(second)
    for key, one in first_variables.items():
        if not one.trainable:
            np.testing.assert_array_equal(
                second_variables[key].numpy(), one.numpy(), err_msg=key[0]
            )

    # outputs: with the weights copied, the rebuilt model is the same function
    for key, one in first_variables.items():
        if one.trainable:
            second_variables[key].assign(one)
    atoms = cluster(first)
    e1, f1 = energy_and_forces(first, atoms)
    e2, f2 = energy_and_forces(second, atoms)
    assert np.abs(f1).max() > 1e-6, "the structure must be sensitive to the weights"
    np.testing.assert_allclose(
        e2, e1, rtol=FLOAT64_ARITHMETIC.rtol, atol=FLOAT64_ARITHMETIC.atol
    )
    np.testing.assert_allclose(
        f2, f1, rtol=FLOAT64_ARITHMETIC.rtol, atol=FLOAT64_ARITHMETIC.atol
    )


# ----------------------------------------------------------------------------------- round trip: presets


@pytest.mark.parametrize("name", list(PRESET_CASES))
def test_preset_round_trips_through_model_yaml(name, tmp_path):
    first = build_model(preset_instructions(name))
    seed_trainable_variables(first)

    assert_round_trip(first, tmp_path / "model.yaml", tmp_path)


@pytest.mark.parametrize("name", list(PRESET_CASES))
def test_saved_yaml_keeps_the_arguments_the_preset_passed(name, tmp_path):
    # hand-written oracle: the element map and the layer sizes given to the preset are in the file
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), preset_instructions(name), param_dtype=tf.float64)
    saved = saved_instructions(path)

    embeddings = [
        d for d in saved.values() if d["__cls__"].endswith(".ScalarChemicalEmbedding")
    ]
    assert len(embeddings) == 1
    assert embeddings[0]["element_map"] == ELEMENT_MAP
    assert embeddings[0]["embedding_size"] == PRESET_CASES[name]["embedding_size"]


@pytest.mark.parametrize("name", list(PRESET_CASES))
def test_saved_keys_are_parameters_of_the_class(name, tmp_path):
    # every saved key is a constructor parameter, so load_instructions can pass it back
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), preset_instructions(name), param_dtype=tf.float64)

    for ins_name, dct in saved_instructions(path).items():
        cls = str_to_class(dct["__cls__"])
        original = signatures.captured_init(cls)
        assert original is not None, ins_name
        parameters = inspect.signature(original).parameters
        has_var_keyword = any(p.kind is p.VAR_KEYWORD for p in parameters.values())
        unknown = set(dct) - {"__cls__"} - set(parameters)
        assert has_var_keyword or not unknown, f"{ins_name}: {sorted(unknown)}"


# --------------------------------------------------------------------------- round trip: yamls of tests/


@pytest.mark.parametrize("filename", TEST_YAMLS)
def test_test_yaml_round_trips_through_model_yaml(filename, tmp_path):
    first = build_model(load_instructions(str(TESTS_DIR / filename)))
    seed_trainable_variables(first)

    assert_round_trip(first, tmp_path / "model.yaml", tmp_path)


@pytest.mark.parametrize("filename", TEST_YAMLS)
def test_resaving_an_old_yaml_adds_the_current_defaults(filename, tmp_path):
    # PINNED, not endorsed: a key that the old file lacks is written with the CURRENT default of the class
    # (the persisted-API gotcha seen from the file side); the keys it has are kept unchanged.
    source = saved_instructions(TESTS_DIR / filename)
    path = tmp_path / "model.yaml"
    save_instructions_dict(
        str(path), load_instructions(str(TESTS_DIR / filename)), param_dtype=tf.float64
    )
    saved = saved_instructions(path)

    assert list(saved) == list(source)
    for name, old in source.items():
        new = saved[name]
        assert all(new[key] == value for key, value in old.items()), name
        defaults = signatures.captured_defaults(str_to_class(old["__cls__"]))
        assert defaults is not None
        for key in set(new) - set(old):
            # dense_nbr=None is resolved against the instruction manager when the instruction is built (False here)
            expected = False if key == "dense_nbr" else _yamlable(defaults[key])
            assert new[key] == expected, f"{name}.{key}"


def _yamlable(value: Any) -> Any:
    """What a default looks like after it went through a yaml dump and load."""
    return yaml.safe_load(yaml.dump(value))


def test_old_yamls_gain_keys_when_resaved(tmp_path):
    # the size of the effect on the 2-layer yaml, counted by class on the file: lm_first on 19 instructions
    # (6 FCRight2Left, 6 ProductFunction, 4 FunctionReduceN, 2 SingleParticleBasisFunctionScalarInd, 1 equivariant
    # SPBF), normalize on 10 and dense_nbr on the equivariant SPBF YI
    source = saved_instructions(TESTS_DIR / "model_grace_2L_omat.yaml")
    path = tmp_path / "model.yaml"
    save_instructions_dict(
        str(path), load_instructions(str(TESTS_DIR / "model_grace_2L_omat.yaml"))
    )
    saved = saved_instructions(path)

    added = {
        key: [n for n in source if key in saved[n] and key not in source[n]]
        for key in ("lm_first", "normalize", "dense_nbr")
    }
    assert {key: len(names) for key, names in added.items()} == {
        "lm_first": 19,
        "normalize": 10,
        "dense_nbr": 1,
    }
    assert added["dense_nbr"] == ["YI"]
    assert saved["YI"]["dense_nbr"] is False
    assert saved["A"]["lm_first"] is False
    assert saved["B"]["normalize"] is True


def test_list_yaml_and_its_resaved_dict_yaml_give_the_same_variables_in_another_order(
    tmp_path,
):
    # PINNED (TEST8 finding): a very old list-format yaml builds a model whose instructions are a list; saving
    # it writes the dict format, and the reloaded dict-based model creates the same variables in another order
    # (tf.Module lists a dict by key, a list by position). This is why update_model converts checkpoints too.
    first = build_model(load_instructions(str(TESTS_DIR / "model_grace.yaml")))
    path = tmp_path / "model.yaml"
    save_instructions_dict(str(path), first.instructions, param_dtype=tf.float64)
    second = build_model(load_instructions(str(path)))

    assert isinstance(first.instructions, list)
    assert isinstance(second.instructions, dict)
    assert variable_specs(first) == variable_specs(second)
    assert [v.name for v in first.variables] != [v.name for v in second.variables]
    assert sorted(v.name for v in first.variables) == sorted(
        v.name for v in second.variables
    )


# --------------------------------------------------------------------------------------- str_to_class


def test_str_to_class_none_gives_none():
    assert str_to_class(None) is None


def test_str_to_class_resolves_the_path_of_a_class():
    from tensorpotential.instructions.compute import BondLength

    assert str_to_class("tensorpotential.instructions.compute.BondLength") is BondLength
    assert class_to_str(BondLength) == "tensorpotential.instructions.compute.BondLength"


def test_str_to_class_name_without_module_is_an_error():
    with pytest.raises(ValueError, match="Couldn't deserialize class `BondLength`"):
        str_to_class("BondLength")


def test_str_to_class_unknown_class_in_a_known_module():
    with pytest.raises(ImportError, match="cannot import name 'NoSuchInstruction'"):
        str_to_class("tensorpotential.instructions.compute.NoSuchInstruction")


def test_str_to_class_unknown_module():
    with pytest.raises(ModuleNotFoundError, match="no_such_package"):
        str_to_class("tensorpotential.no_such_package.BondLength")


def test_str_to_class_reaches_a_class_through_the_tpmodel_getattr_shim():
    # yamls written before the move to tensorpotential.uq store the old path of this class
    from tensorpotential.uq.instructions import RandomProjectedBasisFeatures

    cls = str_to_class("tensorpotential.tpmodel.RandomProjectedBasisFeatures")

    assert cls is RandomProjectedBasisFeatures
    assert issubclass(cls, TPInstruction)


def test_str_to_class_reaches_an_object_through_the_instructions_getattr_shim():
    from tensorpotential.metadata_utils import read_model_metadata

    with pytest.warns(
        DeprecationWarning, match="tensorpotential.metadata_utils.read_model_metadata"
    ):
        resolved = str_to_class("tensorpotential.instructions.read_model_metadata")

    assert resolved is read_model_metadata


@pytest.mark.xfail(
    strict=True,
    reason="PINNED (TEST8 finding): str_to_class builds source text from the string and exec()s it, so a "
    "__cls__ value can carry statements; it should reject anything that is not a dotted path. "
    "Remove the marker when it does.",
)
def test_str_to_class_rejects_a_string_that_is_not_a_dotted_path():
    # harmless payload: it rebinds the result to the builtin len instead of importing os.path.join
    with pytest.raises((ValueError, ImportError, SyntaxError)):
        str_to_class("os.path import join as EXTRACTED_C; EXTRACTED_C = len #.join")


def test_str_to_class_import_that_binds_nothing_is_an_error():
    # reachable only through the statement injection above: the source runs but never binds EXTRACTED_C
    with pytest.raises(ValueError, match="Couldn't deserialize class"):
        str_to_class("os.path import join as other #.join")


def test_every_class_of_the_table_is_reached_by_its_own_path():
    for path, cls in signatures.instruction_classes().items():
        assert class_to_str(cls) == path
        assert str_to_class(path) is cls


# ------------------------------------------------------------------------------- load_instructions


def small_preset_dicts() -> dict[str, dict]:
    """Serialised dicts of a tiny LINEAR model, as ``to_dict`` writes them."""
    preset = presets.LINEAR(element_map=ELEMENT_MAP, **PRESET_CASES["LINEAR"])
    return {name: ins.to_dict() for name, ins in preset.get_instructions().items()}


def write_yaml(path: Path, data: Any) -> str:
    path.write_text(yaml.dump(data, sort_keys=False))
    return str(path)


def test_load_wrapped_format_returns_a_dict_in_file_order(tmp_path):
    dicts = small_preset_dicts()
    path = write_yaml(
        tmp_path / "m.yaml",
        {"metadata": {"param_dtype": "float64"}, "instructions": dicts},
    )

    loaded = load_instructions(path)

    assert isinstance(loaded, dict)
    assert list(loaded) == list(dicts)
    assert [class_to_str(type(i)) for i in loaded.values()] == [
        d["__cls__"] for d in dicts.values()
    ]


def test_load_old_flat_dict_format_drops_the_metadata_sentinel(tmp_path):
    dicts = small_preset_dicts()
    path = write_yaml(
        tmp_path / "m.yaml", {"__metadata__": {"param_dtype": "float64"}, **dicts}
    )

    loaded = load_instructions(path)

    assert isinstance(loaded, dict)
    assert list(loaded) == list(dicts)


def test_load_very_old_list_format_returns_a_list(tmp_path):
    dicts = small_preset_dicts()
    path = write_yaml(tmp_path / "m.yaml", list(dicts.values()))

    loaded = load_instructions(path)

    assert isinstance(loaded, list)
    assert [i.name for i in loaded] == list(dicts)


def test_load_list_format_of_the_yaml_in_tests():
    loaded = load_instructions(str(TESTS_DIR / "model_grace.yaml"))

    assert isinstance(loaded, list)
    assert len(loaded) == 18
    assert len({i.name for i in loaded}) == 18


def test_extract_instructions_data_formats():
    first = {"__cls__": "tensorpotential.instructions.compute.BondLength", "name": "x"}
    second = {
        "__cls__": "tensorpotential.instructions.compute.RadialBasis",
        "name": "y",
    }

    assert _extract_instructions_data({
        "metadata": {},
        "instructions": {"x": first, "y": second},
    }) == (
        [first, second],
        False,
    )
    assert _extract_instructions_data({
        "__metadata__": {},
        "x": first,
        "y": second,
    }) == ([first, second], False)
    assert _extract_instructions_data([first, second]) == ([first, second], True)


def test_load_resolves_references_to_the_instruction_objects_loaded_before(tmp_path):
    # in the LINEAR preset AA = A x A and AAA = AA x A: the references are names in the file, objects after loading
    path = write_yaml(tmp_path / "m.yaml", {"instructions": small_preset_dicts()})

    loaded = load_instructions(path)

    assert loaded["AA"].left is loaded["A"]
    assert loaded["AA"].right is loaded["A"]
    assert loaded["AAA"].left is loaded["AA"]
    assert loaded["AAA"].right is loaded["A"]
    assert loaded["R"].basis_name == "RadialBasis"


def test_load_reference_to_a_later_instruction_is_a_key_error(tmp_path):
    dicts = small_preset_dicts()
    moved_to_the_end = {name: dct for name, dct in dicts.items() if name != "A"}
    moved_to_the_end["A"] = dicts["A"]
    path = write_yaml(tmp_path / "m.yaml", {"instructions": moved_to_the_end})

    with pytest.raises(KeyError, match="'A'"):
        load_instructions(path)


def test_load_bad_constructor_argument_is_logged_and_raised(tmp_path, caplog):
    dicts = small_preset_dicts()
    name = next(iter(dicts))
    dicts[name]["no_such_argument"] = 1
    path = write_yaml(tmp_path / "m.yaml", {"instructions": dicts})

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(TypeError, match="no_such_argument"),
    ):
        load_instructions(path)

    assert "Can't load instruction from" in caplog.text


def test_load_unknown_class_is_an_import_error(tmp_path):
    dicts = small_preset_dicts()
    name = next(iter(dicts))
    dicts[name]["__cls__"] = "tensorpotential.instructions.compute.NoSuchInstruction"
    path = write_yaml(tmp_path / "m.yaml", {"instructions": dicts})

    with pytest.raises(ImportError, match="NoSuchInstruction"):
        load_instructions(path)


def test_load_missing_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_instructions(str(tmp_path / "missing.yaml"))


def test_load_entry_without_a_class_cannot_instantiate_the_abstract_base(tmp_path):
    path = write_yaml(tmp_path / "m.yaml", {"instructions": {"x": {"name": "x"}}})

    with pytest.raises(TypeError):
        load_instructions(path)


# ------------------------------------------------------------------------------------- capture_init_args


@capture_init_args
class Captured:
    """A plain class (not an instruction) with positional, defaulted and keyword-only parameters."""

    def __init__(self, first, second=2, *, third=3, fourth=None):
        self.values = (first, second, third, fourth)


@capture_init_args
class CapturedChild(Captured):
    def __init__(self, first, extra="e", **kwargs):
        super().__init__(first, **kwargs)
        self.extra = extra


def test_capture_init_args_records_defaults_positional_and_keyword_arguments():
    obj = Captured(10, third=30)

    assert obj._init_args == {"first": 10, "second": 2, "third": 30, "fourth": None}
    assert obj.values == (10, 2, 30, None)


def test_capture_init_args_loses_a_positional_argument_given_for_a_defaulted_parameter():
    # PINNED, not endorsed (TEST8 finding): only the parameters without a default are matched to positional
    # arguments, so Captured(1, 5) is built with second=5 but records (and would save) second=2
    obj = Captured(1, 5)

    assert obj.values == (1, 5, 3, None)
    assert obj._init_args == {"first": 1, "second": 2, "third": 3, "fourth": None}


def test_capture_init_args_keyword_only_defaults_come_from_kwdefaults():
    # the keyword-only defaults are in __kwdefaults__, not in the signature defaults tuple
    assert Captured.__init__.__closure__ is not None
    assert signatures.captured_defaults(Captured) == {
        "second": 2,
        "third": 3,
        "fourth": None,
    }


def test_capture_init_args_to_dict_starts_with_the_class_path():
    dct = Captured(1).to_dict()

    assert next(iter(dct)) == "__cls__"
    assert dct == {
        "__cls__": "tests.test_base.Captured",
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": None,
    }


def test_capture_init_args_from_dict_rebuilds_through_the_class_path():
    rebuilt = Captured.from_dict({
        "__cls__": "tests.test_base.Captured",
        "first": 7,
        "third": 9,
    })

    assert isinstance(rebuilt, Captured)
    assert rebuilt.values == (7, 2, 9, None)


def test_capture_init_args_from_dict_without_a_class_path_uses_the_class_it_is_called_on():
    rebuilt = Captured.from_dict({"first": 7})

    assert type(rebuilt) is Captured
    assert rebuilt.values == (7, 2, 3, None)


def test_capture_init_args_from_dict_does_not_modify_its_argument():
    dct = {"__cls__": "tests.test_base.Captured", "first": 7}

    Captured.from_dict(dct)

    assert dct == {"__cls__": "tests.test_base.Captured", "first": 7}


def test_capture_init_args_of_a_child_merges_with_the_arguments_of_the_parent():
    # the child's wrapper runs first, then the parent's: the parent's defaults are added, the child's win
    child = CapturedChild(1, third=5)

    assert child._init_args == {
        "first": 1,
        "extra": "e",
        "second": 2,
        "third": 5,
        "fourth": None,
    }
    assert child.values == (1, 2, 5, None)
    assert child.extra == "e"


# ------------------------------------------------------------------ save_instructions_dict and metadata


def test_save_writes_metadata_then_instructions(tmp_path):
    path = tmp_path / "m.yaml"
    instructions = preset_instructions("LINEAR")

    save_instructions_dict(str(path), instructions, param_dtype=tf.float32)

    with open(path) as f:
        data = yaml.safe_load(f)
    assert list(data) == ["metadata", "instructions"]
    assert data["metadata"]["param_dtype"] == "float32"
    assert data["metadata"]["tensorpotential_version"] == __version__
    assert list(data["instructions"]) == list(instructions)


def test_save_accepts_a_list_and_a_dict_alike(tmp_path):
    instructions = preset_instructions("LINEAR")
    as_dict, as_list = tmp_path / "d.yaml", tmp_path / "l.yaml"

    save_instructions_dict(str(as_dict), instructions)
    save_instructions_dict(str(as_list), list(instructions.values()))

    assert saved_instructions(as_dict) == saved_instructions(as_list)


def test_save_without_param_dtype_writes_no_param_dtype_key(tmp_path):
    path = tmp_path / "m.yaml"

    save_instructions_dict(str(path), preset_instructions("LINEAR"))

    with open(path) as f:
        metadata = yaml.safe_load(f)["metadata"]
    assert set(metadata) == {"tensorpotential_version", "saved_at"}


def test_saved_at_is_an_utc_timestamp():
    from datetime import datetime

    stamp = _build_metadata()["saved_at"]

    assert stamp.endswith("Z")
    datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (tf.float32, "float32"),
        (tf.float64, "float64"),
        # PINNED, not endorsed (TEST8 finding): any dtype other than float32 is written as float64
        (tf.float16, "float64"),
    ],
)
def test_build_metadata_param_dtype_names(dtype, expected):
    assert _build_metadata(dtype)["param_dtype"] == expected


def test_build_metadata_without_dtype_has_version_only_besides_the_timestamp():
    assert _build_metadata() == {
        "tensorpotential_version": __version__,
        "saved_at": _build_metadata()["saved_at"],
    }


# ------------------------------------------------------------------- table of constructor defaults


def test_constructor_defaults_table_is_current():
    problems = signatures.compare_tables(
        signatures.load_table(), signatures.build_table()
    )

    assert not problems, signatures.PERSISTED_API_NOTE + "\n" + "\n".join(problems)


def test_table_covers_the_known_instruction_classes():
    table = signatures.load_table()

    assert len(table) == 39
    assert "tensorpotential.instructions.base.TPInstruction" in table
    assert (
        "tensorpotential.instructions.compute.SingleParticleBasisFunctionEquivariantInd"
        in table
    )
    assert "tensorpotential.instructions.output.LinMLPOut2ScalarTarget" in table
    assert "tensorpotential.uq.instructions.RandomProjectedBasisFeatures" in table


def test_every_captured_default_is_the_signature_default():
    # what the wrapper writes into the yaml is the signature default (and the keyword-only ones)
    for path, cls in signatures.instruction_classes().items():
        original = signatures.captured_init(cls)
        if original is None:
            continue
        expected = {
            name: p.default
            for name, p in inspect.signature(original).parameters.items()
            if name != "self" and p.default is not inspect.Parameter.empty
        }
        captured = signatures.captured_defaults(cls)
        assert captured is not None
        assert captured.keys() == expected.keys(), path
        for name, value in expected.items():
            assert repr(captured[name]) == repr(value), f"{path}.{name}"


def test_only_the_equivariant_base_class_is_not_captured():
    # PINNED (TEST8 finding): TPEquivariantInstruction.__init__ is not decorated; its subclasses are
    table = signatures.load_table()

    assert sorted(path for path, row in table.items() if not row["captured"]) == [
        "tensorpotential.instructions.base.TPEquivariantInstruction"
    ]


def test_table_has_no_object_addresses():
    # a repr with an address changes at every run and would make the table fail by itself
    for path, row in signatures.load_table().items():
        for parameter in row["parameters"]:
            assert " at 0x" not in parameter.get("default_repr", ""), (
                f"{path}.{parameter['name']}"
            )


class TestCompareTables:
    """The comparison reports each kind of change with a sentence naming the class and the parameter."""

    @pytest.fixture
    def committed(self):
        return {
            "pkg.A": {
                "captured": True,
                "parameters": [
                    {
                        "name": "rcut",
                        "kind": "POSITIONAL_OR_KEYWORD",
                        "default_repr": "5.0",
                    },
                    {
                        "name": "name",
                        "kind": "POSITIONAL_OR_KEYWORD",
                        "default_repr": "'A'",
                    },
                    {"name": "bonds", "kind": "POSITIONAL_OR_KEYWORD"},
                ],
            },
        }

    def clone(self, table):
        import copy

        return copy.deepcopy(table)

    def test_equal_tables_have_no_problems(self, committed):
        assert signatures.compare_tables(committed, self.clone(committed)) == []

    def test_a_changed_default_is_reported(self, committed):
        current = self.clone(committed)
        current["pkg.A"]["parameters"][0]["default_repr"] = "6.0"

        assert signatures.compare_tables(committed, current) == [
            "pkg.A: default of 'rcut' changed from 5.0 to 6.0"
        ]

    def test_a_default_that_appears_or_disappears_is_reported(self, committed):
        current = self.clone(committed)
        del current["pkg.A"]["parameters"][0]["default_repr"]
        current["pkg.A"]["parameters"][2]["default_repr"] = "None"

        assert signatures.compare_tables(committed, current) == [
            "pkg.A: default of 'bonds' changed from <none> to None",
            "pkg.A: default of 'rcut' changed from 5.0 to <none>",
        ]

    def test_an_added_and_a_removed_parameter_are_reported(self, committed):
        current = self.clone(committed)
        current["pkg.A"]["parameters"][1] = {
            "name": "label",
            "kind": "POSITIONAL_OR_KEYWORD",
        }

        assert signatures.compare_tables(committed, current) == [
            "pkg.A: parameter 'label' was added",
            "pkg.A: parameter 'name' was removed",
        ]

    def test_a_new_class_without_a_row_is_reported(self, committed):
        current = self.clone(committed)
        current["pkg.B"] = {"captured": True, "parameters": []}

        assert signatures.compare_tables(committed, current) == [
            "pkg.B: class has no row in the table (a new TPInstruction subclass)"
        ]

    def test_a_removed_class_is_reported(self, committed):
        assert signatures.compare_tables(committed, {}) == [
            "pkg.A: class was removed or renamed"
        ]

    def test_capture_added_or_removed_is_reported(self, committed):
        current = self.clone(committed)
        current["pkg.A"]["captured"] = False

        assert signatures.compare_tables(committed, current) == [
            "pkg.A: capture_init_args was removed"
        ]
        assert signatures.compare_tables(current, committed) == [
            "pkg.A: capture_init_args was added"
        ]

    def test_a_changed_kind_and_a_reordering_are_reported(self, committed):
        kind = self.clone(committed)
        kind["pkg.A"]["parameters"][2]["kind"] = "KEYWORD_ONLY"
        order = self.clone(committed)
        order["pkg.A"]["parameters"].reverse()

        assert signatures.compare_tables(committed, kind) == [
            "pkg.A: parameter 'bonds' changed kind from POSITIONAL_OR_KEYWORD to KEYWORD_ONLY"
        ]
        assert signatures.compare_tables(committed, order) == [
            "pkg.A: the order of the parameters changed (positional use breaks)"
        ]


def test_table_main_reports_and_exit_codes(capsys):
    assert signatures.main([]) == 0
    assert "table is current" in capsys.readouterr().out
