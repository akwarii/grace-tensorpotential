"""Twin of TF's ``RadialBasis``: one of the four radial bases of ``ops/radial.py`` on the bond lengths.

Spec sheet ``RadialBasis.md``. ``basis_type`` picks the basis (``Cheb``, ``RadSinBessel``, ``SBessel``, ``Gaussian``);
the basis reads the keywords of the yaml it needs, and the twin evaluates it on the lengths ``[bonds, 1]`` with the
stored cutoff ``rc``, giving ``[bonds, nfunc]`` in float64 (the op owns the zero-length substitution, the strict
``r > rcut`` test and the casts of section 6).

The weights are buffers loaded from the checkpoint (rule R6), never recomputed: ``rc`` for every basis (in the
parameter dtype, so a float32 model evaluates the basis at ``float32(rcut)``), and ``grid`` and ``scale`` for the
Gaussian basis. A keyword the chosen basis does not read must keep its pinned default (owner decision, section 7): a
misspelt or misplaced option is refused rather than ignored. The Gaussian basis also accepts ``rmin``, ``init_gamma``
and ``trainable``, which only set the stored ``grid`` and ``scale``.

Layer: instructions; imports torch, the radial op, the dtype policy, the errors and the pinned defaults of the spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import torch

from tensorpotential.torch_backend.errors import OptionValueError
from tensorpotential.torch_backend.ops.radial import Gaussian, basis_class
from tensorpotential.torch_backend.spec.options import COMPUTE, DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionSpec

_READS: Final[Mapping[str, tuple[str, ...]]] = {
    "Cheb": ("nfunc", "rcut", "p", "kind", "reversed", "normalized"),
    "RadSinBessel": ("nfunc", "rcut", "p", "normalized"),
    "SBessel": ("nfunc", "rcut"),
    "Gaussian": ("nfunc", "rcut", "p"),
}
"""The keywords each basis reads, in the order of its constructor."""
_IGNORED: Final[Mapping[str, tuple[str, ...]]] = {
    "Gaussian": ("rmin", "init_gamma", "trainable")
}
"""Keywords TF reads but the twin does not need: they only set weights that the checkpoint holds."""
_OWN: Final = ("bonds", "basis_type")


def _keywords(spec: InstructionSpec, basis_type: str) -> dict[str, Any]:
    """Return the keywords the basis reads; refuse any other one that is not at its pinned default."""
    defaults = DEFAULTS[COMPUTE + "RadialBasis"]
    reads = _READS[basis_type]
    allowed = (*reads, *_OWN, *_IGNORED.get(basis_type, ()))
    for key, value in spec.options.items():
        if key not in allowed and (key not in defaults or value != defaults[key]):
            msg = (
                f"instruction {spec.name!r} (RadialBasis): the {basis_type} basis does not read {key}={value!r}; "
                f"it reads {', '.join(reads)}"
            )
            raise OptionValueError(msg)
    return {k: spec.options[k] for k in reads if k in spec.options}


class RadialBasis(torch.nn.Module):
    """``RadialBasis``: the radial basis ``[bonds, nfunc]`` of the bond lengths.

    Parameters
    ----------
    spec
        The instruction (``bonds``, ``basis_type`` and the keywords of the basis).
    policy
        Dtype policy of the model: ``rc``, ``grid`` and ``scale`` have its parameter dtype. They start from ``rcut``
        and zeros; the stored values are loaded into them.

    Raises
    ------
    OptionValueError
        For an unknown ``basis_type``, a keyword the basis does not read away from its default, and the values the
        basis refuses (``nfunc``, ``rcut``, ``kind``).
    """

    def __init__(self, spec: InstructionSpec, policy: DtypePolicy) -> None:
        """Build the basis and its weights."""
        super().__init__()
        options = spec.options
        self.spec = spec
        basis_type = options["basis_type"]
        cls = basis_class(basis_type)
        self.basis = cls(**_keywords(spec, basis_type))
        self._lengths = options["bonds"].name
        cpu = torch.device("cpu")
        self.register_buffer("rc", policy.param_constant(float(options["rcut"]), cpu))
        if isinstance(self.basis, Gaussian):
            nfunc = self.basis.nfunc
            self.register_buffer("grid", torch.zeros((1, nfunc), dtype=policy.param))
            self.register_buffer("scale", torch.zeros((), dtype=policy.param))

    @property
    def nfunc(self) -> int:
        """Width of the output."""
        return self.basis.nfunc

    def cutoff(self) -> float:
        """Return the stored cutoff as a Python float: TF's ``get_cutoff`` of a built model (the neighbour-list cutoff)."""
        return float(self.get_buffer("rc"))

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the basis ``[bonds, nfunc]`` at the lengths ``[bonds, 1]``, zero beyond ``rcut``."""
        r, rc = inputs[self._lengths], self.get_buffer("rc")
        if isinstance(self.basis, Gaussian):
            return self.basis(r, rc, self.get_buffer("grid"), self.get_buffer("scale"))
        return self.basis(r, rc)
