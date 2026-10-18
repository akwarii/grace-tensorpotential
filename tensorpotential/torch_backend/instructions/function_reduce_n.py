"""Twin of TF's ``FunctionReduceN``: the learned linear reduction of several basis tensors to one (naive port).

Spec sheet ``FunctionReduceN.md``, section 5, in the backend's layout ``[atoms, lm, n]``:

.. code-block:: text

    out[a, row, k] = norm_map[row] * (init + sum over sources s, and collected entries w of s with
                     total_sum_ind[s][w] = row, of norm_s * sum_n reducing_s[(mu_a,) k, n, w_l_tile[s][w]]
                                                             * source_s[a, func_collect_ind[s][w], n])

``init`` is 0, or 1 with ``init_target_value = "ones"``; ``norm_map`` is 1 unless ``out_norm``. The tables come from
:func:`~tensorpotential.torch_backend.plans.function_reduce_n.function_reduce_n_plan`. When every ``ls_max`` is 0
(``only_invar``) all collected entries are summed into a single row, as TF does whatever the output table says. Each
source is cast to the weight dtype before the contraction, so the output has the parameter dtype. With per-element
weights the sources are cut to the length of ``atomic_mu_i``. The weights ``reducing_<source>`` are the only state
(``state_dict``); the index tables and norms are not saved. Memory: the gathered per-atom weight is
``[atoms, n_out, n_in, n_collected]``, as in TF; a leaner contraction is issue PERF2.

Layer: instructions; imports torch, the plans, the ops, the spec, the errors and the input keys of ``constants``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

from tensorpotential.constants import ATOMIC_MU_I
from tensorpotential.torch_backend.errors import OptionValueError, ShapeError, SpecError
from tensorpotential.torch_backend.ops.precision import DtypePolicy, to_dtype_of
from tensorpotential.torch_backend.plans.function_reduce_n import (
    CollectorPlan,
    FunctionReduceNOptions,
    ReduceSource,
    function_reduce_n_plan,
)
from tensorpotential.torch_backend.spec.model import InstructionRef

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tensorpotential.torch_backend.spec.model import InstructionSpec

INIT_TARGET_VALUES: Final[Mapping[str, float]] = {"zeros": 0.0, "ones": 1.0}
"""``init_target_value``: the value every output entry starts from (it is not a training option, sheet section 5)."""


def _source_names(spec: InstructionSpec) -> tuple[str, ...]:
    refs = spec.options["instructions"]
    return tuple(r.name for r in refs if isinstance(r, InstructionRef))


def _init_value(spec: InstructionSpec) -> float:
    """Return the start value of the output; refuse ``simplify = True`` (decided: rejected, sheet section 7)."""
    if spec.options["simplify"]:
        msg = (
            f"instruction {spec.name!r} (FunctionReduceN): simplify = true is not supported (it rewrites the "
            "coupling histories, and TF fails with it on a ProductFunction source); write simplify: false"
        )
        raise OptionValueError(msg)
    value = spec.options["init_target_value"]
    if value not in INIT_TARGET_VALUES:
        msg = (
            f"instruction {spec.name!r} (FunctionReduceN): init_target_value {value!r} is not one of "
            f"{', '.join(INIT_TARGET_VALUES)}"
        )
        raise OptionValueError(msg)
    return INIT_TARGET_VALUES[value]


class FunctionReduceN(torch.nn.Module):
    """``FunctionReduceN``: select, mix and sum the entries of several sources into one tensor ``[atoms, lm, n_out]``.

    Parameters
    ----------
    spec
        The instruction, as the spec loader read it (``instructions``, ``ls_max``, ``allowed_l_p``, ``n_out``,
        ``out_norm``, ``is_central_atom_type_dependent``, ``number_of_atom_types``, ``normalize``, ``scale``,
        ``init_target_value``; ``lm_first`` is a TF layout and changes nothing here).
    sources
        The coupling table and channel count of each instruction of ``spec.options["instructions"]``, in that order.
    policy
        Dtype policy of the model: the weights have its parameter dtype. They start at zero; the stored weights are
        loaded into ``reducing_<source>``.

    Raises
    ------
    SpecError
        When ``sources`` does not name the instructions of the spec, in order.
    OptionValueError
        For ``simplify = True`` or an unknown ``init_target_value``, and from the plan (distinct sources, one
        ``ls_max`` per source, ``number_of_atom_types`` with per-element weights).
    ShapeError
        From the plan, when a source has no entry to collect.
    """

    def __init__(
        self,
        spec: InstructionSpec,
        sources: Sequence[ReduceSource],
        policy: DtypePolicy,
    ) -> None:
        """Build the plan, the zero weights and the index tables of the instruction."""
        super().__init__()
        given, expected = tuple(s.name for s in sources), _source_names(spec)
        if given != expected:
            msg = (
                f"instruction {spec.name!r} (FunctionReduceN) collects {list(expected)}, "
                f"but the sources given are {list(given)}"
            )
            raise SpecError(msg)
        options = FunctionReduceNOptions.from_options(spec.options)
        self.spec = spec
        self.plan = function_reduce_n_plan(sources, options)
        self.n_out = options.n_out
        self.init_value = _init_value(spec)
        self.per_element = options.is_central_atom_type_dependent
        self.rows = 1 if self.plan.only_invar else len(self.plan.coupling_meta_data)
        self._n_lm = {s.name: len(s.coupling_meta_data) for s in sources}
        for c in self.plan.collectors:
            self.register_parameter(
                f"reducing_{c.name}",
                torch.nn.Parameter(torch.zeros(c.weight_shape, dtype=policy.param)),
            )
            self._index(f"collect_{c.name}", c.func_collect_ind)
            self._index(f"tile_{c.name}", c.w_l_tile)
            self._index(f"sum_{c.name}", c.total_sum_ind.ravel())
        norm_map = self.plan.runtime_norm_map.reshape(1, -1, 1)
        self.register_buffer(
            "norm_map",
            policy.param_constant(norm_map, torch.device("cpu")),
            persistent=False,
        )

    def _index(self, name: str, table: object) -> None:
        index = torch.as_tensor(table, dtype=torch.int64)
        self.register_buffer(name, index, persistent=False)

    def _contribution(
        self, c: CollectorPlan, source: torch.Tensor, atomic_mu_i: torch.Tensor
    ) -> torch.Tensor:
        """``norm_s * w . A`` of one source: ``[atoms, n_collected, n_out]`` in the weight dtype."""
        if source.dim() != 3 or source.shape[1] != self._n_lm[c.name]:
            msg = (
                f"instruction {self.spec.name!r} (FunctionReduceN): source {c.name!r} must be "
                f"[atoms, {self._n_lm[c.name]}, n], got {tuple(source.shape)}"
            )
            raise ShapeError(msg)
        if self.per_element:
            source = source[: atomic_mu_i.shape[0]]
        w = self.get_parameter(f"reducing_{c.name}")
        a = to_dtype_of(source.index_select(1, self.get_buffer(f"collect_{c.name}")), w)
        w = w.index_select(-1, self.get_buffer(f"tile_{c.name}"))
        if self.per_element:
            pr = torch.einsum("aknw,awn->awk", w.index_select(0, atomic_mu_i), a)
        else:
            pr = torch.einsum("knw,awn->awk", w, a)
        return pr * c.norm

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Reduce the sources.

        Parameters
        ----------
        inputs
            Each source ``[atoms, lm, n_in]`` by name, and ``atomic_mu_i`` ``[atoms]`` (the element index of each
            atom; with per-element weights each in ``[0, number_of_atom_types)``).

        Returns
        -------
        torch.Tensor
            ``[atoms, n_rows, n_out]`` in the parameter dtype; ``n_rows`` is the length of the output table, or 1
            with ``only_invar`` (times the rows of ``norm_map`` with ``out_norm``, as TF broadcasts).
        """
        atomic_mu_i = inputs[ATOMIC_MU_I]
        norm_map = self.get_buffer("norm_map")
        acc = torch.full(
            (atomic_mu_i.shape[0], self.rows, self.n_out),
            self.init_value,
            dtype=norm_map.dtype,
            device=norm_map.device,
        )
        for c in self.plan.collectors:
            pr = self._contribution(c, inputs[c.name], atomic_mu_i)
            if self.plan.only_invar:
                acc = acc + pr.sum(dim=1, keepdim=True)
            else:
                acc = acc.index_add(1, self.get_buffer(f"sum_{c.name}"), pr)
        return acc * norm_map if self.plan.out_norm else acc
