"""The check that a twin ignores an initialisation-only option (``OptionRule.init_only``, SPEC6).

TF reads such an option only when it creates a variable; a converted model takes every variable from the checkpoint.
So, for every value the rule accepts, the twin must have the same state (names, shapes, dtypes) as with the default
value and, once given the same weights, the same output to the bit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from tensorpotential.torch_backend.spec.options import SUPPORTED_OPTIONS


def init_only_values(cls: str, option: str) -> tuple[Any, ...]:
    """Return every value the rule of ``option`` of ``cls`` accepts; fail when the option is not initialisation-only.

    Raises
    ------
    ValueError
        When the rule is not marked ``init_only`` or lists no values.
    """
    rule = SUPPORTED_OPTIONS[cls][option]
    if not rule.init_only or not rule.values:
        msg = f"{cls}.{option} is not an initialisation-only option with listed values"
        raise ValueError(msg)
    return rule.values


def _layout(twin: torch.nn.Module) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {k: (tuple(v.shape), v.dtype) for k, v in twin.state_dict().items()}


def assert_option_is_ignored(
    make: Callable[..., torch.nn.Module],
    cls: str,
    option: str,
    inputs: Mapping[str, torch.Tensor],
) -> None:
    """Assert that ``make(**{option: value})`` equals ``make()`` in state and output for every accepted value.

    Parameters
    ----------
    make
        Builds the twin; its keyword arguments are options of the spec.
    cls, option
        The dotted class path and the option, whose accepted values are tried.
    inputs
        The input dictionary of the twin's forward pass.
    """
    reference = make()
    with torch.no_grad():
        expected = reference(inputs)
    for value in init_only_values(cls, option):
        twin = make(**{option: value})
        assert _layout(twin) == _layout(reference), f"{option}={value!r}"
        twin.load_state_dict(reference.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(
                twin(inputs), expected, rtol=0, atol=0, msg=f"{option}={value!r}"
            )
