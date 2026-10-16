"""Radial bases of the ``RadialBasis`` twin: the torch counterpart of ``functions/radial.py``.

Four bases, as the spec sheet ``docs/torch_backend/spec/RadialBasis.md`` describes them (section 5), each a frozen
dataclass of the numbers the yaml gives and a ``__call__`` on the bond lengths and the stored weights:

* :class:`Chebyshev` (``basis_type: Cheb``, ``ChebSqrRadialBasisFunction``): Chebyshev polynomials of the first or second
  kind of ``2 (1 - |1 - r / rc|) - 1`` (degrees 1 to ``nfunc``) times the envelope;
* :class:`SineBessel` (``RadSinBessel``, ``SinBesselRadialBasisFunction``): ``sqrt(2 / rc) sin(n pi r / rc) / r`` times the
  envelope, or its standardised form;
* :class:`SimplifiedBessel` (``SBessel``, ``SimplifiedBesselRadialBasisFunction``): the orthonormal combinations of
  spherical Bessel functions, no envelope;
* :class:`Gaussian` (``GaussianRadialBasisFunction``): Gaussians on a stored grid times the envelope.

Every basis does what ``RadialBasisFunction.__call__`` does around it: a length of exactly 0 is replaced by
``0 + epsilon`` (``1e-10``, ``1e-8`` for the sine Bessel basis), and the result is 0 where the length is strictly larger
than ``rcut``, the Python float of the yaml. The envelope (:func:`cutoff_envelope`) and its first two derivatives
vanish at ``r = rc``. One deliberate difference: TF computes ``|1 - r / rc| ** 1`` in the Chebyshev argument, whose
second derivative is NaN exactly at ``r = rc``; here the power is left out and the second derivative is finite there
(only a Hessian at a length of exactly ``rc`` sees it; values and forces are the same).

Dtypes (section 6 of the sheet). The lengths are geometry (float64); ``rc``, ``grid`` and ``scale`` are weights in the
parameter dtype and are read from the checkpoint, never recomputed from ``rcut`` (rule R6). Where TF combines a
float32 weight with a float64 value it promotes, and the twin does so explicitly (:func:`promote_like_tf`); a
sub-expression of a weight alone keeps its dtype, as in TF (``sqrt(2 / rc)`` of the sine Bessel basis and ``rc**1.5``
of the simplified Bessel basis are float32 for a float32 model). For a cutoff that float32 does not represent
(``rcut = 5.6``) the envelope is then evaluated at ``r / float32(rc)``, and a length equal to ``rcut`` gives a few
1e-15, not 0, as in TF.

Layer: ops; imports torch, numpy, scipy, the errors and the dtype policy (:mod:`.precision`).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np
import torch
from scipy.special import sici

from tensorpotential.torch_backend.errors import OptionValueError, ShapeError
from tensorpotential.torch_backend.ops.precision import (
    DtypePolicy,
    promote_like_tf,
    to_dtype_of,
)

if TYPE_CHECKING:
    from collections.abc import Callable

EPSILON: Final = 1e-10
"""What a length of exactly 0 becomes (``RadialBasisFunction.build``)."""
SINE_BESSEL_EPSILON: Final = 1e-8
"""The same for the sine Bessel basis, whose ``build`` sets its own."""
CHEBYSHEV_NORM: Final = math.sqrt(1.0 / math.pi)
"""The factor of a normalised Chebyshev basis."""
_SQRT2: Final = math.sqrt(2.0)


def cutoff_envelope(x: torch.Tensor, p: int) -> torch.Tensor:
    """Return the polynomial envelope ``E(x; p)`` of order ``p`` (``cutoff_func_p_order_poly``).

    ``E(x) = 1 - (p + 1)(p + 2)/2 x**p + p (p + 2) x**(p + 1) - p (p + 1)/2 x**(p + 2)``: ``E(0) = 1``, and ``E``, its first
    and its second derivative vanish at ``x = 1``.

    Parameters
    ----------
    x
        ``r / rc``, any shape and floating dtype (kept in the result).
    p
        Order of the envelope.
    """
    return (
        1
        - (p + 1) * (p + 2) / 2 * x**p
        + p * (p + 2) * x ** (p + 1)
        - p * (p + 1) / 2 * x ** (p + 2)
    )


def chebyshev_polynomials(u: torch.Tensor, degree: int, kind: int) -> torch.Tensor:
    """Return the Chebyshev polynomials of degrees ``0 .. degree - 1`` at ``u`` (TF's ``chebvander``).

    ``T_0 = 1``, ``T_1 = u`` (first kind) or ``2 u`` (second kind), ``T_k = 2 u T_{k-1} - T_{k-2}``.

    Parameters
    ----------
    u
        ``[n, 1]``.
    degree
        Number of polynomials, at least 1.
    kind
        1 or 2.

    Returns
    -------
    torch.Tensor
        ``[n, degree]``, in the dtype of ``u``.

    Raises
    ------
    OptionValueError
        When ``kind`` is not 1 or 2, or ``degree`` is below 1.
    """
    if kind not in (1, 2):
        msg = f"the Chebyshev kind must be 1 or 2, got {kind!r}"
        raise OptionValueError(msg)
    if degree < 1:
        msg = f"the number of Chebyshev polynomials must be at least 1, got {degree}"
        raise OptionValueError(msg)
    columns = [torch.ones_like(u)]
    if degree > 1:
        columns.append(u if kind == 1 else 2 * u)
    u2 = 2 * u
    for k in range(2, degree):
        columns.append(columns[k - 1] * u2 - columns[k - 2])
    return torch.cat(columns, dim=1)


def _check_lengths(r: torch.Tensor) -> None:
    if r.dim() != 2 or r.shape[1] != 1:
        msg = f"the bond lengths must have the shape [n_bonds, 1], got {tuple(r.shape)}"
        raise ShapeError(msg)


def _check_common(nfunc: int, rcut: float) -> None:
    if nfunc < 1:
        msg = f"a radial basis needs at least one function, got nfunc={nfunc}"
        raise OptionValueError(msg)
    if rcut <= 0:
        msg = f"the cutoff must be larger than 0, got rcut={rcut}"
        raise OptionValueError(msg)


def _on_lengths(
    r: torch.Tensor,
    rc: torch.Tensor,
    epsilon: float,
    rcut: float,
    basis: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """``RadialBasisFunction.__call__``: replace a zero length, evaluate, zero beyond ``rcut`` (strict ``>``)."""
    _check_lengths(r)
    eps = DtypePolicy(rc.dtype).geometry_constant(epsilon, r.device)
    r, eps = promote_like_tf(r, eps)
    r = torch.where(r == 0.0, r + eps, r)
    values = basis(r)
    return torch.where(r > rcut, torch.zeros_like(values), values)


@dataclass(frozen=True)
class Chebyshev:
    """The Chebyshev basis (``basis_type: Cheb``).

    Parameters
    ----------
    nfunc
        Number of functions (degrees 1 to ``nfunc``).
    rcut
        Cutoff of the yaml, for the strict ``r > rcut`` test.
    p
        Order of the envelope.
    kind
        Chebyshev polynomials of the first (1) or second (2) kind. TF raises for another value at the first forward
        pass; this class at construction.
    reversed
        ``True`` flips the sign of the argument.
    normalized
        ``True`` multiplies the basis by :data:`CHEBYSHEV_NORM`.

    Raises
    ------
    OptionValueError
        For ``nfunc < 1``, ``rcut <= 0`` or a ``kind`` other than 1 and 2.
    """

    nfunc: int
    rcut: float
    p: int = 5
    kind: int = 1
    reversed: bool = False
    normalized: bool = False
    epsilon: ClassVar[float] = EPSILON

    def __post_init__(self) -> None:
        """Refuse values TF refuses."""
        _check_common(self.nfunc, self.rcut)
        if self.kind not in (1, 2):
            msg = f"the Chebyshev kind must be 1 or 2, got {self.kind!r}"
            raise OptionValueError(msg)

    def __call__(self, r: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        """Return the basis ``[n_bonds, nfunc]`` at the lengths ``r`` ``[n_bonds, 1]``, ``rc`` the stored cutoff."""
        return _on_lengths(r, rc, self.epsilon, self.rcut, lambda s: self._basis(s, rc))

    def _basis(self, r: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        x = r / to_dtype_of(rc, r)
        if self.reversed:
            u = -2.0 * (1.0 - torch.abs(1.0 - x)) + 1.0
        else:
            u = 2.0 * (1.0 - torch.abs(1.0 - x)) - 1.0
        basis = chebyshev_polynomials(u, self.nfunc + 1, self.kind)[:, 1:]
        basis = basis * cutoff_envelope(x, self.p)
        if self.normalized:
            return basis * CHEBYSHEV_NORM
        return basis


@functools.cache
def sine_bessel_moments(nfunc: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``mu[n]`` and ``sigma[n]``, ``n = 1 .. nfunc``, of the normalised sine Bessel basis (spec sheet, section 2).

    ``mu[n] = sqrt(2) Si(n pi)`` and ``sigma[n]**2 = 2 n pi Si(2 n pi) + cos(2 n pi) - 1 - 2**(3/2) mu[n] Si(n pi) + mu[n]**2``
    (``Si`` the sine integral): the mean and the standard deviation of ``sqrt(2) sin(n pi x) / x`` over ``x`` in
    ``[0, 1]``, computed as the constructor of ``SinBesselRadialBasisFunction`` does.
    """
    npi = np.arange(1, nfunc + 1) * np.pi
    mu = _SQRT2 * sici(npi)[0]
    sigma2 = (  # TF's expression with its unit cutoff urcut = 1.0, whose divisions and products are exact
        2 * npi * sici(2 * npi)[0]
        + (np.cos(2 * npi) - 1)
        - 2 ** (3 / 2) * mu * sici(npi)[0]
        + mu**2
    )
    return tuple(mu.tolist()), tuple(np.sqrt(sigma2).tolist())


@dataclass(frozen=True)
class SineBessel:
    """The sine Bessel basis (``basis_type: RadSinBessel``).

    ``sqrt(2 / rc) sin(n pi r / rc) / r * E(r / rc; p)`` for ``n = 1 .. nfunc``; ``normalized`` gives
    ``(sqrt(2) sin(n pi s) / s - mu[n]) / sigma[n] * E(s; p)`` with ``s = r / rc`` (:func:`sine_bessel_moments`).

    Parameters
    ----------
    nfunc, rcut, p, normalized
        As for :class:`Chebyshev`; ``p`` has no default here, as in TF.

    Raises
    ------
    OptionValueError
        For ``nfunc < 1`` or ``rcut <= 0``.
    """

    nfunc: int
    rcut: float
    p: int
    normalized: bool = False
    epsilon: ClassVar[float] = SINE_BESSEL_EPSILON

    def __post_init__(self) -> None:
        """Refuse values TF refuses."""
        _check_common(self.nfunc, self.rcut)

    def __call__(self, r: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        """Return the basis ``[n_bonds, nfunc]`` at the lengths ``r`` ``[n_bonds, 1]``, ``rc`` the stored cutoff."""
        return _on_lengths(r, rc, self.epsilon, self.rcut, lambda s: self._basis(s, rc))

    def _basis(self, r: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        policy = DtypePolicy(rc.dtype)
        n = torch.arange(1, 1 + self.nfunc, dtype=r.dtype, device=r.device)
        if self.normalized:
            mu, sigma = sine_bessel_moments(self.nfunc)
            s = r / promote_like_tf(rc, r)[0]
            f = (
                _SQRT2 * torch.sin(n * s * np.pi) / s
                - policy.geometry_constant(mu, r.device)
            ) / policy.geometry_constant(sigma, r.device)
            return f * cutoff_envelope(s, self.p)
        rc64 = promote_like_tf(rc, r)[0]
        # sqrt(2 / rc) in the dtype of rc, as in TF
        prefactor = promote_like_tf(torch.sqrt(2 / rc), r)[0]
        return (
            prefactor
            * torch.sin(n * r * np.pi / rc64)
            / r
            * cutoff_envelope(r / rc64, self.p)
        )


@functools.cache
def simplified_bessel_recursion(
    nfunc: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``e[n]`` and ``d[n]``, ``n = 0 .. nfunc - 1``, of the orthonormalising recursion of the simplified Bessel basis.

    ``e_n = n**2 (n + 2)**2 / (4 (n + 1)**4 + 1)``, ``d_0 = 1``, ``d_n = 1 - e_n / d_{n-1}`` (``e_0`` is not used).
    """
    n = np.arange(nfunc, dtype=np.float64)
    e = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
    d = np.ones(nfunc)
    for i in range(1, nfunc):
        d[i] = 1 - e[i] / d[i - 1]
    return tuple(e.tolist()), tuple(d.tolist())


def _sinc(x: torch.Tensor) -> torch.Tensor:
    """``sin(x) / x`` (``torch.sinc`` is the normalised ``sin(pi x) / (pi x)``).

    TF's ``sinc`` also returns 1 at ``x = 0``; that branch is never taken, because a zero length is replaced by
    ``epsilon`` before a basis is evaluated, so it is left out.
    """
    return torch.sin(x) / x


@dataclass(frozen=True)
class SimplifiedBessel:
    """The simplified spherical Bessel basis (``basis_type: SBessel``); it has no envelope.

    With ``f_n(r) = (-1)**n sqrt(2) pi / rc**(3/2) (n+1)(n+2) / sqrt((n+1)**2 + (n+2)**2) (sinc((n+1) pi r / rc) +
    sinc((n+2) pi r / rc))``: ``B_0 = f_0`` and ``B_n = (f_n + sqrt(e_n / d_{n-1}) B_{n-1}) / sqrt(d_n)``
    (:func:`simplified_bessel_recursion`), the functions orthonormal with the weight ``r**2`` on ``[0, rc]``.

    Parameters
    ----------
    nfunc, rcut
        As for :class:`Chebyshev`.

    Raises
    ------
    OptionValueError
        For ``nfunc < 1`` or ``rcut <= 0``.
    """

    nfunc: int
    rcut: float
    epsilon: ClassVar[float] = EPSILON

    def __post_init__(self) -> None:
        """Refuse values TF refuses."""
        _check_common(self.nfunc, self.rcut)

    def __call__(self, r: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        """Return the basis ``[n_bonds, nfunc]`` at the lengths ``r`` ``[n_bonds, 1]``, ``rc`` the stored cutoff."""
        return _on_lengths(r, rc, self.epsilon, self.rcut, lambda s: self._basis(s, rc))

    def _basis(self, r: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        # rc**1.5 in the dtype of rc, as in TF
        rc64, rc_power = promote_like_tf(rc, rc**1.5, r)[:2]

        def f(n: int) -> torch.Tensor:
            prefactor = (
                (-1.0) ** n
                * _SQRT2
                * np.pi
                / rc_power
                * (n + 1)
                * (n + 2)
                / math.sqrt((n + 1) ** 2 + (n + 2) ** 2)
            )
            return prefactor * (
                _sinc(r * (n + 1) * np.pi / rc64) + _sinc(r * (n + 2) * np.pi / rc64)
            )

        e, d = simplified_bessel_recursion(self.nfunc)
        columns = [f(0)]
        for i in range(1, self.nfunc):
            columns.append(
                1
                / math.sqrt(d[i])
                * (f(i) + math.sqrt(e[i] / d[i - 1]) * columns[i - 1])
            )
        return torch.cat(columns, dim=1)


@dataclass(frozen=True)
class Gaussian:
    """The Gaussian basis: ``exp(scale (r - grid)**2) E(r / rc; p)``, ``grid`` and ``scale`` stored weights.

    Parameters
    ----------
    nfunc, rcut
        As for :class:`Chebyshev`; at least two functions (TF's constructor fails for one).
    p
        Order of the envelope.

    Raises
    ------
    OptionValueError
        For ``nfunc < 2`` or ``rcut <= 0``.
    """

    nfunc: int
    rcut: float
    p: int
    epsilon: ClassVar[float] = EPSILON

    def __post_init__(self) -> None:
        """Refuse values TF cannot build."""
        _check_common(self.nfunc, self.rcut)
        if self.nfunc < 2:  # TF spaces the centres by grid[1] - grid[0]
            msg = (
                f"a Gaussian basis needs at least two functions, got nfunc={self.nfunc}"
            )
            raise OptionValueError(msg)

    def __call__(
        self,
        r: torch.Tensor,
        rc: torch.Tensor,
        grid: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Return the basis ``[n_bonds, nfunc]``; ``grid`` ``[1, nfunc]`` and ``scale`` ``[]`` are the stored weights.

        Raises
        ------
        ShapeError
            When ``grid`` or ``scale`` does not have its shape.
        """
        if tuple(grid.shape) != (1, self.nfunc) or scale.dim() != 0:
            msg = (
                f"a Gaussian basis of {self.nfunc} functions needs grid [1, {self.nfunc}] and a scalar scale, "
                f"got {tuple(grid.shape)} and {tuple(scale.shape)}"
            )
            raise ShapeError(msg)
        return _on_lengths(
            r, rc, self.epsilon, self.rcut, lambda s: self._basis(s, rc, grid, scale)
        )

    def _basis(
        self,
        r: torch.Tensor,
        rc: torch.Tensor,
        grid: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        rc64, grid64, scale64 = promote_like_tf(rc, grid, scale, r)[:3]
        return torch.exp(scale64 * (r - grid64) ** 2) * cutoff_envelope(
            r / rc64, self.p
        )


BASES: Final = {
    "Cheb": Chebyshev,
    "RadSinBessel": SineBessel,
    "SBessel": SimplifiedBessel,
    "Gaussian": Gaussian,
}
"""The class of each ``basis_type`` that ``RadialBasis`` accepts (``compute.py:384-393``)."""


def basis_class(
    basis_type: str,
) -> type[Chebyshev | SineBessel | SimplifiedBessel | Gaussian]:
    """Return the class of a ``basis_type``.

    Raises
    ------
    OptionValueError
        For a name that TF does not know, naming the four it does.
    """
    try:
        return BASES[basis_type]
    except KeyError:
        known = ", ".join(BASES)
        msg = f"unknown radial basis_type {basis_type!r}; the known ones are {known}"
        raise OptionValueError(msg) from None
