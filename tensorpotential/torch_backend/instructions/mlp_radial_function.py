"""Twin of TF's ``MLPRadialFunction``: the radial functions ``R_nl(r)`` as a bias-free MLP of the radial basis.

Spec sheet ``MLPRadialFunction.md``, section 5. With ``b`` the basis ``[bonds, nfunc]``:

.. code-block:: text

    b = concat([b, Z_i[mu_i], Z_j[mu_j]])          (each embedding only when the yaml names one; cast up to b's dtype)
    y = mlp(b)                                     ([bonds, n_rad_max * (lmax + 1)]; activation on the hidden layers)
    y = y * rsqrt(var(y) + 1e-5) * gamma           (only with norm)
    R[bond, n, l*l + l + m] = y[bond, n * (lmax + 1) + l]

TF writes ``R`` as ``[bonds, n_rad_max, (lmax + 1)**2]``; the twin returns it in the backend layout
``[bonds, (lmax + 1)**2, n_rad_max]``, the radial input of ``SingleParticleBasisFunction*``. The first layer casts the
basis to the parameter dtype, so a float32 model gives a float32 ``R`` from a float64 basis, as in TF. The weights are
``mlp.layer<i>.w`` (and ``gamma`` with ``norm``), as in the checkpoint.

Layer: instructions; imports torch, the dense ops, the dtype helpers, the errors, the column table of the spherical
harmonics and the input keys of ``constants``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorpotential.constants import BOND_MU_I, BOND_MU_J
from tensorpotential.torch_backend.errors import OptionValueError
from tensorpotential.torch_backend.ops.dense import MLP, variance_norm
from tensorpotential.torch_backend.ops.precision import check_index, to_dtype_of
from tensorpotential.torch_backend.plans.spherical_harmonics import column_degrees

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec

# hidden_layers of an MLPRadialFunction whose yaml gives null (compute.py)
_DEFAULT_HIDDEN_LAYERS = (64, 64, 64)


class MLPRadialFunction(torch.nn.Module):
    """``MLPRadialFunction``: ``R [bonds, (lmax + 1)**2, n_rad_max]`` from the radial basis.

    Parameters
    ----------
    spec
        The instruction (``basis``, ``n_rad_max``, ``lmax``, ``hidden_layers``, ``norm``, ``activation``,
        ``chemical_embedding_i``, ``chemical_embedding_j``; ``no_weight_decay`` and ``input_shape`` do not change the
        forward pass).
    basis_size
        Width of the basis input (TF's ``nfunc`` of the basis instruction).
    policy
        Dtype policy of the model: the weights and ``gamma`` have its parameter dtype.
    embedding_size_i, embedding_size_j
        Width of the ``chemical_embedding_i`` / ``chemical_embedding_j`` table, ``None`` without one.

    Raises
    ------
    OptionValueError
        For a null ``basis``, ``n_rad_max`` below 1 or a negative ``lmax``, an embedding size given without its
        embedding or the reverse, and from the MLP (activation).
    """

    def __init__(
        self,
        spec: InstructionSpec,
        basis_size: int,
        policy: DtypePolicy,
        *,
        embedding_size_i: int | None = None,
        embedding_size_j: int | None = None,
    ) -> None:
        """Build the MLP, ``gamma`` and the column table."""
        super().__init__()
        options = spec.options
        self.spec = spec
        # the option table refuses a null basis too; TF fails only at the first forward pass
        if options["basis"] is None:
            msg = f"instruction {spec.name!r} (MLPRadialFunction): basis is null"
            raise OptionValueError(msg)
        self._basis = options["basis"].name
        self.n_rad_max, self.lmax = int(options["n_rad_max"]), int(options["lmax"])
        if self.n_rad_max < 1 or self.lmax < 0:
            msg = f"instruction {spec.name!r} (MLPRadialFunction): n_rad_max must be at least 1 and lmax at least 0, got {self.n_rad_max} and {self.lmax}"
            raise OptionValueError(msg)
        sides = (("i", embedding_size_i), ("j", embedding_size_j))
        embeddings = [
            _embedding(spec, side, options[f"chemical_embedding_{side}"], size)
            for side, size in sides
        ]
        self._embeddings = tuple(e for e in embeddings if e is not None)
        n_in = basis_size + sum(size for *_, size in self._embeddings)
        hidden = options["hidden_layers"]
        hidden = _DEFAULT_HIDDEN_LAYERS if hidden is None else tuple(hidden)
        n_out = self.n_rad_max * (self.lmax + 1)
        self.mlp = MLP.fully_connected(
            [n_in, *hidden, n_out], options["activation"], policy
        )
        self.gamma = None
        if options["norm"]:
            self.gamma = torch.nn.Parameter(torch.zeros((1, n_out), dtype=policy.param))
        l_tile = torch.as_tensor(column_degrees(self.lmax), dtype=torch.int64)
        self.register_buffer("l_tile", l_tile, persistent=False)

    def _input(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the basis, with the embedding rows of the central atom and then the neighbour appended."""
        basis = inputs[self._basis]
        columns = [basis]
        for name, index_key, _ in self._embeddings:
            index = inputs[index_key]
            check_index(index)
            table = to_dtype_of(inputs[name], basis)
            columns.append(table.index_select(0, index))
        return torch.cat(columns, dim=1) if len(columns) > 1 else basis

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return ``R [bonds, (lmax + 1)**2, n_rad_max]`` in the parameter dtype.

        Parameters
        ----------
        inputs
            The basis ``[bonds, nfunc]`` by name; with an embedding also its table ``[n_elements, size]`` by name and
            ``mu_i`` / ``mu_j`` ``[bonds]``.
        """
        y = self.mlp(self._input(inputs))
        if self.gamma is not None:
            y = variance_norm(y, self.gamma)
        # [bonds, n, l] -> [bonds, l, n], then each l spread over its 2 l + 1 columns
        y = y.reshape(-1, self.n_rad_max, self.lmax + 1).transpose(1, 2)
        return y.index_select(1, self.get_buffer("l_tile"))


def _embedding(
    spec: InstructionSpec, side: str, ref: InstructionRef | None, size: int | None
) -> tuple[str, str, int] | None:
    """Return ``(table name, index key, width)`` of one embedding, or ``None`` without one."""
    if (ref is None) != (size is None):
        msg = f"instruction {spec.name!r} (MLPRadialFunction): its chemical_embedding_{side} is {ref!r}, but an embedding size of {size!r} was given"
        raise OptionValueError(msg)
    if ref is None or size is None:
        return None
    return ref.name, BOND_MU_I if side == "i" else BOND_MU_J, size
