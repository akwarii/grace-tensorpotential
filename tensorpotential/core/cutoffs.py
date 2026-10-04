"""Per-pair cutoff tables: the presets and the expansion of a user-written cutoff dictionary.

The expansion is pure Python (no TensorFlow, no NumPy), so both backends share it.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping

PairCutoffs = dict[tuple[str, str], float]

_ELEMENT_SYMBOL = re.compile(r"[A-Z][a-z]*")

CUTOFF_PRESETS = {
    # Equals the former table "equilibrium nn-dist + 3 A, crop(4,8).round(1)" floored at 5.0 A
    # (all 87 elements); the former table is in git history: git show 67d7fa6322f4:tensorpotential/utils.py
    # },
    "DEFAULT_CUTOFF_1L": {
        "H": 5.0,
        "N": 5.0,
        "O": 5.0,
        "C": 5.0,
        "F": 5.0,
        "B": 5.0,
        "Cl": 5.0,
        "S": 5.1,
        "Mn": 5.2,
        "P": 5.2,
        "Be": 5.2,
        "Se": 5.4,
        "Si": 5.4,
        "Br": 5.4,
        "Fe": 5.4,
        "Co": 5.5,
        "Cr": 5.5,
        "Ni": 5.5,
        "Ge": 5.5,
        "Ga": 5.5,
        "As": 5.6,
        "Cu": 5.6,
        "V": 5.6,
        "Ti": 5.6,
        "Zn": 5.7,
        "Ru": 5.7,
        "Os": 5.7,
        "Tc": 5.7,
        "Rh": 5.7,
        "Mo": 5.7,
        "Ir": 5.7,
        "Re": 5.7,
        "W": 5.8,
        "Pd": 5.8,
        "Pt": 5.8,
        "I": 5.8,
        "Al": 5.9,
        "Nb": 5.9,
        "Ta": 5.9,
        "Sn": 5.9,
        "Te": 5.9,
        "He": 5.9,
        "Au": 5.9,
        "Ag": 5.9,
        "Sb": 5.9,
        "Cd": 6.0,
        "Li": 6.0,
        "Ne": 6.0,
        "Bi": 6.1,
        "Hf": 6.1,
        "Mg": 6.2,
        "Zr": 6.2,
        "Sc": 6.2,
        "Po": 6.3,
        "Ce": 6.3,
        "In": 6.4,
        "Lu": 6.4,
        "Tm": 6.4,
        "Er": 6.5,
        "Ho": 6.5,
        "Y": 6.5,
        "Dy": 6.5,
        "Tl": 6.5,
        "Hg": 6.5,
        "Tb": 6.5,
        "Pb": 6.6,
        "Gd": 6.6,
        "Sm": 6.6,
        "Pm": 6.6,
        "Nd": 6.7,
        "Pr": 6.7,
        "La": 6.7,
        "Na": 6.7,
        "Yb": 6.8,
        "Ca": 6.9,
        "Eu": 6.9,
        "Ar": 7.0,
        "Sr": 7.2,
        "Ba": 7.4,
        "Ra": 7.5,
        "Kr": 7.5,
        "K": 7.7,
        "Xe": 7.9,
        "Rb": 8.0,
        "Fr": 8.0,
        "Cs": 8.0,
        "Rn": 8.0,
    },
    # Not derived from the former table "equilibrium nn-dist + 1.75 A, crop(3.5,7.5).round(1)"
    # (differs by 0.1 to 1.5 A); the former table is in git history: git show 67d7fa6322f4:tensorpotential/utils.py
    # },
    "DEFAULT_CUTOFF_2L": {
        "H": 5.0,
        "N": 5.0,
        "O": 5.0,
        "C": 5.0,
        "F": 5.0,
        "B": 5.0,
        "Cl": 5.0,
        "S": 5.0,
        "Mn": 5.0,
        "P": 5.0,
        "Be": 5.0,
        "Se": 5.0,
        "Si": 5.0,
        "Br": 5.0,
        "Fe": 5.0,
        "Co": 5.0,
        "Cr": 5.0,
        "Ni": 5.0,
        "Ge": 5.0,
        "Ga": 5.0,
        "As": 5.0,
        "Cu": 5.0,
        "V": 5.0,
        "Ti": 5.0,
        "Zn": 5.0,
        "Ru": 5.0,
        "Os": 5.0,
        "Tc": 5.0,
        "Rh": 5.0,
        "Mo": 5.0,
        "Ir": 5.0,
        "Re": 5.0,
        "W": 5.0,
        "Pd": 5.0,
        "Pt": 5.1,
        "I": 5.1,
        "Al": 5.1,
        "Nb": 5.1,
        "Ta": 5.1,
        "Sn": 5.1,
        "Te": 5.1,
        "He": 5.1,
        "Au": 5.2,
        "Ag": 5.2,
        "Sb": 5.2,
        "Cd": 5.3,
        "Li": 5.3,
        "Ne": 5.3,
        "Bi": 5.3,
        "Hf": 5.4,
        "Mg": 5.4,
        "Zr": 5.4,
        "Sc": 5.5,
        "Po": 5.5,
        "Ce": 5.6,
        "In": 5.6,
        "Lu": 5.7,
        "Tm": 5.7,
        "Er": 5.7,
        "Ho": 5.8,
        "Y": 5.8,
        "Dy": 5.8,
        "Tl": 5.8,
        "Hg": 5.8,
        "Tb": 5.8,
        "Pb": 5.8,
        "Gd": 5.8,
        "Sm": 5.9,
        "Pm": 5.9,
        "Nd": 5.9,
        "Pr": 6.0,
        "La": 6.0,
        "Na": 6.0,
        "Yb": 6.1,
        "Ca": 6.1,
        "Eu": 6.2,
        "Ar": 6.2,
        "Sr": 6.5,
        "Ba": 6.6,
        "Ra": 6.8,
        "Kr": 6.8,
        "K": 7.0,
        "Xe": 7.1,
        "Rb": 7.3,
        "Fr": 7.5,
        "Cs": 7.5,
        "Rn": 7.5,
    },
    "BIG_CUTOFF_2L": {
        "H": 5.0,
        "N": 5.0,
        "O": 5.0,
        "C": 5.0,
        "F": 5.0,
        "B": 5.0,
        "Cl": 5.0,
        "S": 5.0,
        "Mn": 6.0,
        "P": 5.0,
        "Be": 5.0,
        "Se": 5.0,
        "Si": 5.0,
        "Br": 5.0,
        "Fe": 6.0,
        "Co": 6.0,
        "Cr": 6.0,
        "Ni": 6.0,
        "Ge": 5.0,
        "Ga": 6.0,
        "As": 5.0,
        "Cu": 6.0,
        "V": 6.0,
        "Ti": 6.0,
        "Zn": 6.0,
        "Ru": 6.0,
        "Os": 6.0,
        "Tc": 5.0,
        "Rh": 6.0,
        "Mo": 6.0,
        "Ir": 6.0,
        "Re": 6.0,
        "W": 6.0,
        "Pd": 6.0,
        "Pt": 6.0,
        "I": 5.1,
        "Al": 6.0,
        "Nb": 6.0,
        "Ta": 6.0,
        "Sn": 6.0,
        "Te": 5.1,
        "He": 6.0,
        "Au": 6.0,
        "Ag": 6.0,
        "Sb": 5.2,
        "Cd": 6.0,
        "Li": 6.0,
        "Ne": 6.0,
        "Bi": 6.0,
        "Hf": 6.0,
        "Mg": 6.0,
        "Zr": 6.0,
        "Sc": 6.0,
        "Po": 6.0,
        "Ce": 6.0,
        "In": 6.0,
        "Lu": 6.0,
        "Tm": 6.0,
        "Er": 6.0,
        "Ho": 6.0,
        "Y": 6.0,
        "Dy": 6.0,
        "Tl": 6.0,
        "Hg": 6.0,
        "Tb": 6.0,
        "Pb": 6.0,
        "Gd": 6.0,
        "Sm": 6.0,
        "Pm": 6.0,
        "Nd": 6.0,
        "Pr": 6.0,
        "La": 6.0,
        "Na": 6.0,
        "Yb": 6.1,
        "Ca": 6.1,
        "Eu": 6.2,
        "Ar": 6.2,
        "Sr": 6.5,
        "Ba": 6.6,
        "Ra": 6.8,
        "Kr": 6.8,
        "K": 7.0,
        "Xe": 7.1,
        "Rb": 7.3,
        "Fr": 7.5,
        "Cs": 7.5,
        "Rn": 7.5,
    },
    "CUTOFF_2L": {
        "H": 6.0,
        "N": 6.0,
        "O": 6.0,
        "C": 5.0,
        "F": 6.0,
        "B": 5.5,
        "Cl": 6.0,
        "S": 5.5,
        "Mn": 6.0,
        "P": 5.5,
        "Be": 5.0,
        "Se": 5.5,
        "Si": 5.5,
        "Br": 6.0,
        "Fe": 6.0,
        "Co": 6.0,
        "Cr": 6.0,
        "Ni": 6.0,
        "Ge": 6.0,
        "Ga": 6.0,
        "As": 6.0,
        "Cu": 6.0,
        "V": 6.0,
        "Ti": 6.0,
        "Zn": 6.0,
        "Ru": 6.0,
        "Os": 6.0,
        "Tc": 6.0,
        "Rh": 6.0,
        "Mo": 6.0,
        "Ir": 6.0,
        "Re": 6.0,
        "W": 6.0,
        "Pd": 6.0,
        "Pt": 6.0,
        "I": 6.0,
        "Al": 6.0,
        "Nb": 6.0,
        "Ta": 6.0,
        "Sn": 6.0,
        "Te": 6.0,
        "He": 6.0,
        "Au": 6.0,
        "Ag": 6.0,
        "Sb": 5.5,
        "Cd": 6.0,
        "Li": 6.0,
        "Ne": 6.0,
        "Bi": 6.0,
        "Hf": 6.0,
        "Mg": 6.0,
        "Zr": 6.0,
        "Sc": 6.0,
        "Po": 6.0,
        "Ce": 6.0,
        "In": 6.0,
        "Lu": 6.0,
        "Tm": 6.0,
        "Er": 6.0,
        "Ho": 6.0,
        "Y": 6.0,
        "Dy": 6.0,
        "Tl": 6.0,
        "Hg": 6.0,
        "Tb": 6.0,
        "Pb": 6.0,
        "Gd": 6.0,
        "Sm": 6.0,
        "Pm": 6.0,
        "Nd": 6.0,
        "Pr": 6.0,
        "La": 6.0,
        "Na": 6.0,
        "Yb": 6.1,
        "Ca": 6.1,
        "Eu": 6.2,
        "Ar": 6.2,
        "Sr": 6.5,
        "Ba": 6.6,
        "Ra": 6.8,
        "Kr": 6.8,
        "K": 7.0,
        "Xe": 7.1,
        "Rb": 7.3,
        "Fr": 7.5,
        "Cs": 7.5,
        "Rn": 7.5,
    },
}


def _preset_pair_table(
    radii: Mapping[str, float], elements: Collection[str]
) -> dict[str, float]:
    """Pair cutoffs of a preset: ``r_a`` for a pair of equal elements, ``(r_a + r_b) / 2`` otherwise."""
    kept = {e: r for e, r in radii.items() if e in elements}
    table: dict[tuple[str, str], float] = {}
    for e1, r1 in kept.items():
        table[e1, e1] = r1
        for e2, r2 in kept.items():
            table[e1, e2] = (r1 + r2) / 2
    return {"".join(pair): r for pair, r in table.items()}


def _expand_star(pair_cutoffs: dict, elements: list[str]) -> None:
    """Replace every key containing ``*`` by one key per element, in place.

    The expanded keys are written after the explicit ones, so a ``*`` entry wins over an explicit
    entry for the same pair whatever the order in which they were written.
    """
    expanded = {}
    starred = []
    for key, value in pair_cutoffs.items():
        if "*" in key:
            expanded.update({key.replace("*", str(e)): value for e in elements})
            starred.append(key)
    for key in starred:
        del pair_cutoffs[key]
    pair_cutoffs.update(expanded)


def _as_element_pair(key: str | tuple[str, str]) -> tuple[str, str]:
    """Parse ``"CuAl"`` or ``"Cu"`` (same element twice) or accept a pair of symbols."""
    if isinstance(key, str):
        symbols = tuple(_ELEMENT_SYMBOL.findall(key))
        return (symbols[0], symbols[0]) if len(symbols) == 1 else symbols
    if isinstance(key, tuple):
        if len(key) != 2 or not all(isinstance(symbol, str) for symbol in key):
            msg = f"A cutoff key given as a tuple must hold two element symbols, got {key!r}"
            raise AssertionError(msg)
        return key
    msg = f"Can not parse {key}"
    raise ValueError(msg)


def process_cutoff_dict(pair_cutoff_map: str | dict, element_map: dict) -> PairCutoffs:
    """Expand a user-defined cutoff specification into a table ``(element, element) -> cutoff``.

    Parameters
    ----------
    pair_cutoff_map
        Either the name of a preset in ``CUTOFF_PRESETS`` or a dictionary whose keys are
        ``"CuAl"`` (a pair), ``"Cu"`` (the same element twice), ``"Cu*"`` or ``"*"`` (``*`` stands
        for every element of ``element_map``) or a tuple of two symbols. The dictionary is not
        modified.
    element_map
        Elements of the model (only the keys are used). Pairs with an element outside it are dropped.

    Returns
    -------
    dict
        Both orders of every pair. For two different orders written with different values, each
        order receives the value written for the other one.

    Raises
    ------
    AssertionError
        A tuple key that does not hold two strings.
    ValueError
        A key that is neither a string nor a tuple.
    AttributeError
        A string that is not the name of a preset.

    Examples
    --------
    >>> process_cutoff_dict({"CuAl": 3.5}, {"Cu": 0, "Al": 1})
    {('Cu', 'Al'): 3.5, ('Al', 'Cu'): 3.5}
    """
    elements = sorted(set(element_map))
    if isinstance(pair_cutoff_map, dict):
        spec = pair_cutoff_map.copy()
    elif pair_cutoff_map in CUTOFF_PRESETS:
        spec = _preset_pair_table(CUTOFF_PRESETS[pair_cutoff_map], elements)
    else:
        msg = f"{pair_cutoff_map!r} is not the name of a cutoff preset {sorted(CUTOFF_PRESETS)}"
        raise AttributeError(msg)
    _expand_star(spec, elements)

    table = {_as_element_pair(key): value for key, value in spec.items()}
    table.update({(e2, e1): value for (e1, e2), value in table.items()})
    return {
        pair: value
        for pair, value in table.items()
        if pair[0] in element_map and pair[1] in element_map
    }
