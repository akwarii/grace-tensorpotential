"""Constructor signatures of every ``TPInstruction`` subclass: the table that pins the persisted API.

``capture_init_args`` writes every constructor argument, defaults included, into the ``model.yaml`` of a
saved model, so a default value, a parameter name or a new parameter is part of every model that already
exists. ``tests/data/instruction_constructor_defaults.json`` records them; ``test_base.py`` compares it with
the classes. After a deliberate change, regenerate the table from the repository root::

    python -m tests.instruction_signatures --write

The signature is read from the original ``__init__`` kept in the closure of the ``capture_init_args``
wrapper (``inspect.signature`` of the wrapper only shows ``*args, **kwargs``).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

TABLE_PATH = Path(__file__).parent / "data" / "instruction_constructor_defaults.json"

PERSISTED_API_NOTE = (
    "Constructor arguments and their defaults are persisted API: capture_init_args writes the defaults into every "
    "saved model.yaml, so a changed default silently changes models that already exist, and a renamed, "
    "added or removed parameter breaks loading of existing yamls. If the change is deliberate, regenerate the "
    "table with `python -m tests.instruction_signatures --write` and say so in the commit."
)


def _import_instruction_modules() -> None:
    """Import every module of the package that defines a ``TPInstruction`` subclass."""
    import tensorpotential.instructions  # noqa: F401
    import tensorpotential.tpmodel  # noqa: F401
    import tensorpotential.uq.instructions  # noqa: F401

    try:
        import tensorpotential.extra.gen_tensor.model  # noqa: F401
    except ModuleNotFoundError:
        pass


def _all_subclasses(cls: type) -> list[type]:
    found: list[type] = []
    for sub in cls.__subclasses__():
        found.append(sub)
        found.extend(_all_subclasses(sub))
    return found


def instruction_classes() -> dict[str, type]:
    """The ``TPInstruction`` class and its subclasses defined in the package, by dotted path.

    Classes of ``tensorpotential.compat`` (legacy, out of scope) and classes defined elsewhere (a test
    stub) are not part of the table.
    """
    from tensorpotential.instructions.base import TPInstruction, class_to_str

    _import_instruction_modules()
    classes = {class_to_str(TPInstruction): TPInstruction}
    for cls in _all_subclasses(TPInstruction):
        if cls.__module__.startswith(
            "tensorpotential."
        ) and not cls.__module__.startswith("tensorpotential.compat."):
            classes[class_to_str(cls)] = cls
    return dict(sorted(classes.items()))


def captured_init(cls: type) -> Callable[..., None] | None:
    """The original ``__init__`` of a class decorated with ``capture_init_args``, else ``None``."""
    init = cls.__dict__.get("__init__")
    code = getattr(init, "__code__", None)
    if code is None or "original_init" not in code.co_freevars:
        return None
    return init.__closure__[code.co_freevars.index("original_init")].cell_contents  # type: ignore[index]


def captured_defaults(cls: type) -> dict[str, Any] | None:
    """The defaults the ``capture_init_args`` wrapper of ``cls`` writes into the saved yaml."""
    init = cls.__dict__.get("__init__")
    code = getattr(init, "__code__", None)
    if code is None or "default_values" not in code.co_freevars:
        return None
    return dict(
        init.__closure__[code.co_freevars.index("default_values")].cell_contents
    )  # type: ignore[index]


def _parameter_row(parameter: inspect.Parameter) -> dict[str, str]:
    row = {"name": parameter.name, "kind": parameter.kind.name}
    if parameter.default is not inspect.Parameter.empty:
        row["default_repr"] = repr(parameter.default)
    return row


def class_row(cls: type) -> dict[str, Any]:
    """One row of the table: whether the constructor is captured and its parameters with defaults."""
    original = captured_init(cls)
    decorated = original is not None
    function = original if decorated else cls.__dict__.get("__init__", cls.__init__)
    parameters = [
        _parameter_row(p)
        for name, p in inspect.signature(function).parameters.items()
        if name != "self"
    ]
    return {"captured": decorated, "parameters": parameters}


def build_table() -> dict[str, dict[str, Any]]:
    """The table for the classes of the installed package."""
    return {path: class_row(cls) for path, cls in instruction_classes().items()}


def load_table() -> dict[str, dict[str, Any]]:
    """The committed table."""
    return json.loads(TABLE_PATH.read_text())


def _compare_parameters(path: str, old: list[dict], new: list[dict]) -> list[str]:
    old_by_name = {p["name"]: p for p in old}
    new_by_name = {p["name"]: p for p in new}
    problems = [
        f"{path}: parameter {name!r} was removed"
        for name in old_by_name.keys() - new_by_name.keys()
    ]
    problems += [
        f"{path}: parameter {name!r} was added"
        for name in new_by_name.keys() - old_by_name.keys()
    ]
    for name in old_by_name.keys() & new_by_name.keys():
        before, after = old_by_name[name], new_by_name[name]
        if before.get("default_repr") != after.get("default_repr"):
            problems.append(
                f"{path}: default of {name!r} changed from {before.get('default_repr', '<none>')} "
                f"to {after.get('default_repr', '<none>')}"
            )
        if before["kind"] != after["kind"]:
            problems.append(
                f"{path}: parameter {name!r} changed kind from {before['kind']} to {after['kind']}"
            )
    if not problems and [p["name"] for p in old] != [p["name"] for p in new]:
        problems.append(
            f"{path}: the order of the parameters changed (positional use breaks)"
        )
    return problems


def compare_tables(committed: dict, current: dict) -> list[str]:
    """Differences between the committed table and the classes, one sentence each (empty if equal)."""
    problems = [
        f"{path}: class was removed or renamed"
        for path in committed.keys() - current.keys()
    ]
    problems += [
        f"{path}: class has no row in the table (a new TPInstruction subclass)"
        for path in current.keys() - committed.keys()
    ]
    for path in committed.keys() & current.keys():
        if committed[path]["captured"] != current[path]["captured"]:
            problems.append(
                f"{path}: capture_init_args was {'removed' if committed[path]['captured'] else 'added'}"
            )
        problems += _compare_parameters(
            path, committed[path]["parameters"], current[path]["parameters"]
        )
    return sorted(problems)


def main(argv: list[str] | None = None) -> int:
    """Print the differences with the committed table, or rewrite it with ``--write``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write", action="store_true", help="rewrite the committed table"
    )
    args = parser.parse_args(argv)
    table = build_table()
    if args.write:
        TABLE_PATH.parent.mkdir(exist_ok=True)
        TABLE_PATH.write_text(json.dumps(table, indent=1, sort_keys=False) + "\n")
        sys.stdout.write(f"wrote {len(table)} classes to {TABLE_PATH}\n")
        return 0
    problems = compare_tables(load_table(), table)
    sys.stdout.write("\n".join(problems or ["table is current"]) + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
