"""Option matrix of the TF-trained ``model.yaml`` files the torch twins accept, and the constructor defaults pinned for them.

The data here is a decision, not a measurement: ``tools/scan_options.py`` tabulates every (class, option, kind of
value) of the scanned yamls (``tests/data/model_yamls``, ``tests/model_grace*.yaml``) and ``tests/test_options.py``
asserts that every cell is either supported (or ignored) by :data:`SUPPORTED_OPTIONS`, or rejected with a reason
by :data:`REJECTED_CLASSES` / :data:`REJECTED_OPTIONS`. The policy is closed-world: a value that no scanned,
in-scope model uses has no twin and is rejected, whatever the TF class would accept (rule R3: unsupported means
an error, never a silent approximation).

``DEFAULTS`` repeats the constructor defaults of the TF classes: ``capture_init_args`` writes them into every
saved yaml, so a yaml that omits a key means this value, and a changed TF default would silently change old
models. They are compared with ``tests/data/instruction_constructor_defaults.json`` (and, in SPEC5, with the
source). Kinds are the type names of ``tools/scan_options.value_kind``: ``null``, ``bool``, ``int``, ``float``,
``str``, ``ref`` (an ``_instruction_`` reference), ``list[int]``, ``list[list[int]]``, ``dict[str,int]``.

The module imports neither TensorFlow nor torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

COMPUTE: Final = "tensorpotential.instructions.compute."
OUTPUT: Final = "tensorpotential.instructions.output."


@dataclass(frozen=True)
class OptionRule:
    """What a yaml may hold under one option of one class.

    Parameters
    ----------
    kinds
        Accepted value kinds.
    values
        Exact values accepted for a categorical or numeric option (the elements for ``list[str]``);
        ``None`` accepts any value of the kinds.
    ignored
        The key is swallowed by the ``**kwargs`` of the TF constructor and has no effect on the forward pass;
        it is accepted so that existing yamls load, and nothing reads it.
    """

    kinds: tuple[str, ...]
    values: tuple[Any, ...] | None = None
    ignored: bool = False


@dataclass(frozen=True)
class Rejection:
    """A value seen in a scanned yaml that has no twin, with the reason."""

    kind: str
    reason: str


def _r(
    *kinds: str, values: tuple[Any, ...] | None = None, ignored: bool = False
) -> OptionRule:
    return OptionRule(kinds, values, ignored)


_REF = _r("ref")
_LAYOUT = _r(
    "bool"
)  # lm_first: both layouts are ported (R3); equality is tested with a TF fixture pair.
_NO_LORA = _r("null", values=(None,))
_NULL = _r("null", values=(None,))
_TRUE = _r("bool", values=(True,))
_FALSE = _r("bool", values=(False,))
_BOOL = _r("bool")
_INT = _r("int")
_REAL = _r("float", "int")
_ZERO = _r("int", values=(0,))  # the angular degree `l` of a scalar target
_PARITY = _r("list[list[int]]")

SUPPORTED_OPTIONS: Final[Mapping[str, Mapping[str, OptionRule]]] = {
    COMPUTE + "BondLength": {"instruction_with_bonds": _NULL},
    COMPUTE + "ScaledBondVector": {"bond_length": _REF, "bonds": _NULL},
    COMPUTE + "RadialBasis": {
        "bonds": _REF,
        "basis_type": _r("str", values=("Cheb", "SBessel")),
        "nfunc": _INT,
        "rcut": _REAL,
        "p": _INT,
        "normalized": _BOOL,
        "kind": _r("int", values=(1,)),
        "reversed": _FALSE,
    },
    COMPUTE + "SphericalHarmonic": {"vhat": _REF, "lmax": _INT},
    COMPUTE + "MLPRadialFunction": {
        "n_rad_max": _INT,
        "lmax": _INT,
        "basis": _REF,
        "input_shape": _NULL,
        "hidden_layers": _r("list[int]"),
        "norm": _FALSE,
        "activation": _r("str", values=("tanh",)),
        "no_weight_decay": _TRUE,
        "chemical_embedding_i": _NULL,
        "chemical_embedding_j": _NULL,
        "lora_config": _NO_LORA,
    },
    COMPUTE + "MLPRadialFunction_v2": {
        "n_rad_max": _INT,
        "lmax": _INT,
        "basis": _REF,
        "input_shape": _NULL,
        "hidden_layers": _r("list[int]"),
        "activation": _r("list[str]", values=("silu",)),
        "no_weight_decay": _TRUE,
        "init_type": _r("str", values=("normal",)),
        "normalize": _TRUE,
        "chem_embedding": _NULL,
        "embed_i": _FALSE,
        "embed_j": _TRUE,
        "lora_config": _NO_LORA,
    },
    COMPUTE + "ScalarChemicalEmbedding": {
        "element_map": _r("dict[str,int]"),
        "embedding_size": _INT,
        "is_trainable": _TRUE,
        "init": _r("str", values=("random",)),
        "lora_config": _NO_LORA,
    },
    COMPUTE + "SingleParticleBasisFunctionScalarInd": {
        "radial": _REF,
        "angular": _REF,
        "indicator": _REF,
        "indicator_l_depend": _FALSE,
        "sum_neighbors": _TRUE,
        "avg_n_neigh": _REAL,
        "lora_config": _NO_LORA,
        "lmax": _NULL,
        "lm_first": _LAYOUT,
    },
    COMPUTE + "SingleParticleBasisFunctionEquivariantInd": {
        "angular": _REF,
        "indicator": _REF,
        "lmax": _INT,
        "Lmax": _INT,
        "radial": _REF,
        "keep_parity": _PARITY,
        "history_drop_list": _NULL,
        "l_max_ind": _NULL,
        "max_sum_l": _NULL,
        "sum_neighbors": _TRUE,
        "avg_n_neigh": _REAL,
        "normalize": _TRUE,
        "radial_basis": _NULL,
        "hidden_layers": _NULL,
        "chemical_embedding": _NULL,
        "lm_first": _LAYOUT,
        "dense_nbr": _r("bool", "null"),  # both layouts are ported (R3)
        "n_out": _r("null", values=(None,), ignored=True),
        "radia_basis": _r(
            "null", values=(None,), ignored=True
        ),  # typo of radial_basis, kept in old yamls
    },
    COMPUTE + "ProductFunction": {
        "left": _REF,
        "right": _REF,
        "lmax": _INT,
        "Lmax": _INT,
        "is_left_right_equal": _r("bool", "null", values=(True, None)),
        "lmax_left": _NULL,
        "lmax_right": _NULL,
        "lmax_hist": _NULL,
        "lmax_hist_left": _NULL,
        "lmax_hist_right": _NULL,
        "history_drop_list": _NULL,
        "max_sum_l": _NULL,
        "keep_parity": _PARITY,
        "normalize": _TRUE,
        "lm_first": _LAYOUT,
        "chemical_embedding": _r("null", values=(None,), ignored=True),
        "downscale_embedding_size": _r("int", ignored=True),
        "n_out": _r("null", values=(None,), ignored=True),
    },
    COMPUTE + "FCRight2Left": {
        "left": _REF,
        "right": _REF,
        "n_out": _INT,
        "left_coefs": _BOOL,
        "is_central_atom_type_dependent": _r("bool", "null", values=(False, None)),
        "number_of_atom_types": _NULL,
        "init_vars": _r("str", values=("random",)),
        "normalize": _TRUE,
        "norm_out": _TRUE,
        "lora_config": _NO_LORA,
        "lm_first": _LAYOUT,
    },
    COMPUTE + "FunctionReduceN": {
        "instructions": _r("list[ref]"),
        "ls_max": _r("int", "list[int]"),
        "n_out": _INT,
        "allowed_l_p": _PARITY,
        "out_norm": _BOOL,
        "is_central_atom_type_dependent": _BOOL,
        "number_of_atom_types": _r("int", "null"),
        "init_vars": _r("str", values=("random",)),
        "normalize": _TRUE,
        "init_target_value": _r("str", values=("zeros",)),
        "simplify": _BOOL,
        "scale": _r("float"),
        "lora_config": _NO_LORA,
        "lm_first": _LAYOUT,
        "chemical_embedding": _r("null", values=(None,), ignored=True),
        "downscale_embedding_size": _r("int", ignored=True),
        "n_in": _r("null", values=(None,), ignored=True),
    },
    COMPUTE + "InvariantLayerRMSNorm": {
        "inpt": _REF,
        "type": _r("str", values=("full", "only_nonlin")),
        "init": _r("str", values=("zeros",)),
    },
    OUTPUT + "CreateOutputTarget": {"initial_value": _r("float"), "l": _ZERO},
    OUTPUT + "LinMLPOut2ScalarTarget": {
        "origin": _r("list[ref]"),
        "target": _REF,
        "hidden_layers": _r("list[int]"),
        "n_out": _r("int", values=(1,)),
        "normalize": _r("null", "str", values=(None, "layer")),
        "activation": _r("null", "str", values=(None, "silu", "tanh")),
        "l": _ZERO,
        "lora_config": _NO_LORA,
        "return_hidden_target": _NULL,
        "full_origin_norm": _r("bool", ignored=True),
        "init_norm": _r(
            "str", ignored=True
        ),  # no such parameter in the TF source: swallowed by **kwargs
    },
    OUTPUT + "ConstantScaleShiftTarget": {
        "target": _REF,
        "scale": _REAL,
        "shift": _REAL,
        "atomic_shift_map": _NULL,
        "chemical_embedding": _NULL,
        "l": _ZERO,
    },
    OUTPUT + "TrainableShiftTarget": {
        "target": _REF,
        "number_of_atom_types": _INT,
        "l": _ZERO,
    },
}

_OUTSIDE = (
    "not one of the instruction classes the first twin set covers (decision D2: the 2L OMAT and "
    "large models first); needed by {models}"
)
REJECTED_CLASSES: Final[Mapping[str, str]] = {
    COMPUTE + "BondSpecificRadialBasisFunction": _OUTSIDE.format(
        models="the SMAX models (1L and 2L)"
    ),
    COMPUTE + "SPBF": _OUTSIDE.format(models="the 3L model"),
    COMPUTE + "GeneralProductFunction": _OUTSIDE.format(models="the 3L model"),
    COMPUTE + "EquivariantRMSNorm": _OUTSIDE.format(models="the 3L model"),
    COMPUTE + "LinearRadialFunction": _OUTSIDE.format(models="the FS family"),
    COMPUTE + "CropProductFunction": _OUTSIDE.format(models="the FS family"),
    OUTPUT + "FSOut2ScalarTarget": _OUTSIDE.format(models="the FS family"),
    COMPUTE + "FunctionReduce": _OUTSIDE.format(
        models="the legacy list-format model_grace.yaml"
    ),
    OUTPUT + "LinearOut2Target": _OUTSIDE.format(
        models="the legacy list-format model_grace.yaml"
    ),
    OUTPUT + "MLPOut2ScalarTarget": _OUTSIDE.format(
        models="the legacy list-format model_grace.yaml (and its lm_first=True fails in TF)"
    ),
}

_ONLY_3L = (
    "seen only in GRACE-3L-OMAT-large, which needs classes outside the first twin set"
)
REJECTED_OPTIONS: Final[Mapping[str, Mapping[str, tuple[Rejection, ...]]]] = {
    COMPUTE + "FCRight2Left": {"n_out": (Rejection("null", _ONLY_3L),)},
    COMPUTE + "FunctionReduceN": {
        "n_in": (
            Rejection(
                "list[int]",
                "seen only in GRACE-FS-OMAT (FS family, rejected class set)",
            ),
        )
    },
    COMPUTE + "MLPRadialFunction": {
        "hidden_layers": (
            Rejection(
                "null",
                "seen only in the legacy list-format model_grace.yaml, which the loader rejects",
            ),
        )
    },
    OUTPUT + "ConstantScaleShiftTarget": {
        "atomic_shift_map": (Rejection("dict[int,float]", _ONLY_3L),)
    },
}

LORA_REJECTION: Final = (
    "LoRA adapters are not supported: TensorPotential.enable_lora_adaptation and finalize_lora_update raise "
    "NotImplementedError (tensorpot.py), and none of the scanned yamls holds a non-null lora_config. A non-null "
    "value would add adapter variables the twins do not have."
)

# Model-level (``metadata`` block of the wrapped yaml format) options.
MODEL_OPTIONS: Final[Mapping[str, OptionRule]] = {
    "param_dtype": _r("str", values=("float32", "float64")),
    "tensorpotential_version": _r("str", ignored=True),
}
# A yaml without a metadata block is an old model: metadata_utils.resolve_param_dtype reads it as float64 (the
# value TensorPotential and gracemaker use for a new training run is float32, which does not apply to loading).
MODEL_DEFAULTS: Final[Mapping[str, Any]] = {"param_dtype": "float64"}

_C = COMPUTE
_O = OUTPUT
DEFAULTS: Final[Mapping[str, Mapping[str, Any]]] = {
    _C + "BondLength": {"instruction_with_bonds": None},
    _C + "ScaledBondVector": {"bonds": None},
    # nfunc and rcut have no default; p, normalized, kind and reversed are those of ChebSqrRadialBasisFunction
    _C + "RadialBasis": {"p": 5, "normalized": False, "kind": 1, "reversed": False},
    _C + "MLPRadialFunction": {
        "basis": None,
        "input_shape": None,
        "hidden_layers": None,
        "norm": False,
        "activation": None,
        "no_weight_decay": True,
        "chemical_embedding_i": None,
        "chemical_embedding_j": None,
        "lora_config": None,
    },
    _C + "MLPRadialFunction_v2": {
        "basis": None,
        "input_shape": None,
        "hidden_layers": None,
        "activation": None,
        "no_weight_decay": True,
        "init_type": "normal",
        "normalize": True,
        "chem_embedding": None,
        "embed_i": False,
        "embed_j": True,
        "lora_config": None,
    },
    _C + "ScalarChemicalEmbedding": {
        "is_trainable": True,
        "init": "random",
        "lora_config": None,
    },
    _C + "SingleParticleBasisFunctionScalarInd": {
        "indicator": None,
        "indicator_l_depend": False,
        "sum_neighbors": True,
        "avg_n_neigh": 1.0,
        "lora_config": None,
        "lmax": None,
        "lm_first": False,
    },
    _C + "SingleParticleBasisFunctionEquivariantInd": {
        "keep_parity": None,
        "history_drop_list": None,
        "l_max_ind": None,
        "max_sum_l": None,
        "sum_neighbors": True,
        "avg_n_neigh": 1.0,
        "normalize": False,
        "radial_basis": None,
        "hidden_layers": None,
        "chemical_embedding": None,
        "lm_first": False,
        "dense_nbr": None,
    },
    _C + "ProductFunction": {
        "is_left_right_equal": None,
        "lmax_left": None,
        "lmax_right": None,
        "lmax_hist": None,
        "lmax_hist_left": None,
        "lmax_hist_right": None,
        "history_drop_list": None,
        "max_sum_l": None,
        "keep_parity": None,
        "normalize": False,
        "lm_first": False,
    },
    _C + "FCRight2Left": {
        "n_out": None,
        "left_coefs": True,
        "is_central_atom_type_dependent": None,
        "number_of_atom_types": None,
        "init_vars": "random",
        "normalize": True,
        "norm_out": False,
        "lora_config": None,
        "lm_first": False,
    },
    _C + "FunctionReduceN": {
        "out_norm": False,
        "is_central_atom_type_dependent": False,
        "number_of_atom_types": None,
        "init_vars": "random",
        "normalize": True,
        "init_target_value": "zeros",
        "simplify": False,
        "scale": 1.0,
        "lora_config": None,
        "lm_first": False,
    },
    _C + "InvariantLayerRMSNorm": {"type": "only_nonlin", "init": "zeros"},
    _O + "CreateOutputTarget": {"initial_value": 0.0, "l": 0},
    _O + "LinMLPOut2ScalarTarget": {
        "hidden_layers": None,
        "n_out": 1,
        "normalize": None,
        "activation": None,
        "l": 0,
        "lora_config": None,
        "return_hidden_target": None,
    },
    _O + "ConstantScaleShiftTarget": {
        "scale": 1.0,
        "shift": 0.0,
        "atomic_shift_map": None,
        "chemical_embedding": None,
        "l": 0,
    },
    _O + "TrainableShiftTarget": {"l": 0},
}
