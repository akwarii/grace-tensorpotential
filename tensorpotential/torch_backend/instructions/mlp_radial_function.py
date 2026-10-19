"""Twins of TF's ``MLPRadialFunction`` and ``MLPRadialFunction_v2``: the radial functions ``R_nl(r)`` of the basis.

Spec sheet ``MLPRadialFunction.md``, section 5. With ``b`` the basis ``[bonds, nfunc]``:

.. code-block:: text

    b = concat([b, Z_i[mu_i], Z_j[mu_j]])          (each embedding only when the yaml names one; cast up to b's dtype)
    y = mlp(b)                                     ([bonds, n_rad_max * (lmax + 1)]; activation on the hidden layers)
    y = y * rsqrt(var(y) + 1e-5) * gamma           (only with norm)
    R[bond, n, l*l + l + m] = y[bond, n * (lmax + 1) + l]

TF writes ``R`` as ``[bonds, n_rad_max, (lmax + 1)**2]``; the twin returns it in the backend layout
``[bonds, (lmax + 1)**2, n_rad_max]``, the radial input of ``SingleParticleBasisFunction*``. The first layer casts the
basis to the parameter dtype, so a float32 model gives a float32 ``R`` from a float64 basis, as in TF. The weights are
``mlp.layer<i>.w`` (and ``gamma`` with ``norm``), as in the checkpoint. The embeddings are gathered by the element
indices ``mu_i`` / ``mu_j``: TF's per-atom branch (``is_per_atom``, rejected by the sheet) cannot be reached, since no
embedding option sets the flag.

``MLPRadialFunction_v2`` (sheet ``MLPRadialFunction_v2.md``, section 5) is a bias-free MLP of ``Linear`` layers with one
activation per hidden layer (plain ``silu`` by default) and ``normalize`` deciding whether each layer applies
``1/sqrt(n_in)``; with a ``chem_embedding`` its output is gated by ``z = embed_transform(Z)`` (``[n_elements, n_out]``):

.. code-block:: text

    gate = z[mu_j]                        (embed_j)
    gate = tanh(gate * z[mu_i])           (embed_i; the tanh sits in this branch only, as in TF; 1 without embed_j)
    y = mlp(b) * gate                     (no gate when neither flag is set)

The embedding and the gate stay in the parameter dtype (no cast up, unlike v1). The output layout is that of v1. The
weights are ``layers.<i>.w`` and ``embed_transform.w``, as in the checkpoint. The twins have no ``local`` (domain
decomposed) mode, which TF refuses with a ``chem_embedding`` anyway.

Layer: instructions; imports torch, the dense ops, the dtype helpers, the errors, the column table of the spherical
harmonics and the input keys of ``constants``.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch

from tensorpotential.constants import BOND_MU_I, BOND_MU_J
from tensorpotential.torch_backend.errors import OptionValueError
from tensorpotential.torch_backend.ops.dense import (
    ACTIVATIONS,
    MLP,
    Linear,
    variance_norm,
)
from tensorpotential.torch_backend.ops.precision import check_index, to_dtype_of
from tensorpotential.torch_backend.plans.spherical_harmonics import column_degrees

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorpotential.torch_backend.ops.precision import DtypePolicy
    from tensorpotential.torch_backend.spec.model import InstructionRef, InstructionSpec

# hidden_layers of an MLPRadialFunction whose yaml gives null (compute.py)
_DEFAULT_HIDDEN_LAYERS = (64, 64, 64)


def _radial_sizes(spec: InstructionSpec) -> tuple[int, int]:
    """Return ``(n_rad_max, lmax)`` of a radial MLP; refuse ``n_rad_max < 1`` and ``lmax < 0``."""
    n_rad_max, lmax = int(spec.options["n_rad_max"]), int(spec.options["lmax"])
    if n_rad_max < 1 or lmax < 0:
        cls = spec.cls.rsplit(".", 1)[-1]
        msg = f"instruction {spec.name!r} ({cls}): n_rad_max must be at least 1 and lmax at least 0, got {n_rad_max} and {lmax}"
        raise OptionValueError(msg)
    return n_rad_max, lmax


def _radial_layout(
    y: torch.Tensor, n_rad_max: int, lmax: int, l_tile: torch.Tensor
) -> torch.Tensor:
    """``R[b, l*l + l + m, n] = y[b, n * (lmax + 1) + l]``: the MLP output in the backend layout ``[bonds, lm, n]``."""
    # [bonds, n, l] -> [bonds, l, n], then each l spread over its 2 l + 1 columns
    y = y.reshape(-1, n_rad_max, lmax + 1).transpose(1, 2)
    return y.index_select(1, l_tile)


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
        self.n_rad_max, self.lmax = _radial_sizes(spec)
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
        l_tile = torch.from_numpy(column_degrees(self.lmax))
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
        return torch.cat(columns, dim=1)

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
        return _radial_layout(y, self.n_rad_max, self.lmax, self.get_buffer("l_tile"))


def _embedding(
    spec: InstructionSpec, side: str, ref: InstructionRef | None, size: int | None
) -> tuple[str, str, int] | None:
    """Return ``(table name, index key, width)`` of one embedding, or ``None`` without one."""
    if (ref is None) != (size is None):
        msg = f"instruction {spec.name!r} (MLPRadialFunction): its chemical_embedding_{side} is {ref!r}, but an embedding size of {size!r} was given"
        raise OptionValueError(msg)
    if (
        ref is None or size is None
    ):  # one is None only if both are; both tests narrow the types
        return None
    return ref.name, BOND_MU_I if side == "i" else BOND_MU_J, size


# hidden_layers and activation of an MLPRadialFunction_v2 whose yaml gives null (compute.py)
_DEFAULT_HIDDEN_LAYERS_V2 = (64, 64)
_DEFAULT_ACTIVATION_V2 = "silu"


def _activations_v2(spec: InstructionSpec, n_hidden: int) -> tuple[str, ...]:
    """Return one activation name per hidden layer: null is ``silu``, a string is repeated, a list is taken as it is."""
    given = spec.options["activation"]
    if given is None or isinstance(given, str):
        names = (given or _DEFAULT_ACTIVATION_V2,) * n_hidden
    else:
        names = tuple(given)
    if len(names) != n_hidden or not set(names) <= set(ACTIVATIONS) - {"silu_n2norm"}:
        msg = (
            f"instruction {spec.name!r} (MLPRadialFunction_v2): activation needs one of tanh, silu, sigmoid per hidden "
            f"layer ({n_hidden}), got {given!r}"
        )
        raise OptionValueError(msg)
    return names


class MLPRadialFunction_v2(torch.nn.Module):
    """``MLPRadialFunction_v2``: ``R [bonds, (lmax + 1)**2, n_rad_max]`` from the radial basis, optionally gated.

    Parameters
    ----------
    spec
        The instruction (``basis``, ``n_rad_max``, ``lmax``, ``hidden_layers``, ``activation``, ``normalize``,
        ``chem_embedding``, ``embed_i``, ``embed_j``; ``init_type``, ``no_weight_decay`` and ``input_shape`` do not
        change the forward pass).
    basis_size
        Width of the basis input (TF's ``nfunc`` of the basis instruction).
    policy
        Dtype policy of the model: the weights have its parameter dtype.
    embedding_size
        Width of the ``chem_embedding`` table, ``None`` without one.

    Raises
    ------
    OptionValueError
        For a null ``basis``, ``n_rad_max`` below 1 or a negative ``lmax``, an activation that is not one known name
        per hidden layer, or an embedding size given without its embedding or the reverse.
    """

    def __init__(
        self,
        spec: InstructionSpec,
        basis_size: int,
        policy: DtypePolicy,
        *,
        embedding_size: int | None = None,
    ) -> None:
        """Build the layers, the embedding transform and the column table."""
        super().__init__()
        options = spec.options
        self.spec = spec
        if options["basis"] is None:
            msg = f"instruction {spec.name!r} (MLPRadialFunction_v2): basis is null"
            raise OptionValueError(msg)
        self._basis = options["basis"].name
        self.n_rad_max, self.lmax = _radial_sizes(spec)
        hidden = options["hidden_layers"]
        hidden = _DEFAULT_HIDDEN_LAYERS_V2 if hidden is None else tuple(hidden)
        n_out = self.n_rad_max * (self.lmax + 1)
        sizes = [basis_size, *hidden, n_out]
        self.activations = tuple(
            ACTIVATIONS[a] for a in _activations_v2(spec, len(hidden))
        )
        normalize = bool(options["normalize"])
        self.layers = torch.nn.ModuleList(
            Linear(n_in, n_next, policy, normalize=normalize)
            for n_in, n_next in itertools.pairwise(sizes)
        )
        self._gate = self._build_gate(spec, embedding_size, n_out, policy)
        l_tile = torch.from_numpy(column_degrees(self.lmax))
        self.register_buffer("l_tile", l_tile, persistent=False)

    def _build_gate(
        self,
        spec: InstructionSpec,
        embedding_size: int | None,
        n_out: int,
        policy: DtypePolicy,
    ) -> tuple[str, bool, bool] | None:
        """Make ``embed_transform`` and return ``(table name, embed_i, embed_j)``, or ``None`` without a gate."""
        embedding = spec.options["chem_embedding"]
        if (embedding is None) != (embedding_size is None):
            msg = f"instruction {spec.name!r} (MLPRadialFunction_v2): its chem_embedding is {embedding!r}, but an embedding size of {embedding_size!r} was given"
            raise OptionValueError(msg)
        self.embed_transform = None
        # one is None only if both are; both tests narrow the types
        if embedding is None or embedding_size is None:
            return None
        # always normalised, whatever the option (compute.py)
        self.embed_transform = Linear(embedding_size, n_out, policy)
        return (
            embedding.name,
            bool(spec.options["embed_i"]),
            bool(spec.options["embed_j"]),
        )

    def _gated(
        self, y: torch.Tensor, inputs: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Multiply ``y`` by the gate of the embedding (TF's order: neighbour, then centre and the tanh)."""
        if self._gate is None or self.embed_transform is None:
            return y
        name, embed_i, embed_j = self._gate
        if not (embed_i or embed_j):
            return y
        z = self.embed_transform(inputs[name])
        gate: torch.Tensor | float = 1.0
        if embed_j:
            check_index(inputs[BOND_MU_J])
            gate = z.index_select(0, inputs[BOND_MU_J])
        if embed_i:
            check_index(inputs[BOND_MU_I])
            gate = torch.tanh(gate * z.index_select(0, inputs[BOND_MU_I]))
        return y * gate

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return ``R [bonds, (lmax + 1)**2, n_rad_max]`` in the parameter dtype.

        Parameters
        ----------
        inputs
            The basis ``[bonds, nfunc]`` by name; with a gate also the embedding table ``[n_elements, size]`` by name
            and ``mu_i`` / ``mu_j`` ``[bonds]``.
        """
        x = inputs[self._basis]
        *hidden, last = self.layers
        for layer, act in zip(hidden, self.activations, strict=True):
            x = act(layer(x))
        y = self._gated(last(x), inputs)
        return _radial_layout(y, self.n_rad_max, self.lmax, self.get_buffer("l_tile"))
