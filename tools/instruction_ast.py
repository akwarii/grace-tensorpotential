"""TF-free extractor of the instruction classes and their constructor defaults, read from the source with ``ast``.

TensorFlow is never imported: the extractor parses every ``.py`` file of ``<root>/tensorpotential`` (except
``compat/``, legacy and out of scope) and finds the ``TPInstruction`` subclasses by following the base classes
through the imports of each module. For each class it records the dotted path, whether its own ``__init__`` is
decorated with ``capture_init_args`` (a decorated class writes every constructor argument, defaults included, into the
saved ``model.yaml``; ``capture_init_args`` is a class decorator) and the parameters of the constructor that applies (the class's own, else the nearest
ancestor's: what ``inspect.signature`` reports).

The table has the same shape as ``tests/data/instruction_constructor_defaults.json`` (written by
``tests/instruction_signatures.py`` from the imported classes, so with TensorFlow): the two are independent
oracles of each other, and ``tests/test_instruction_ast.py`` compares them.

Two commands, run from the repository root::

    python tools/instruction_ast.py table  [--root DIR]     # print the table as JSON
    python tools/instruction_ast.py check  [--root DIR]     # compare with the registry; exit 1 on a difference

``check`` is what the scheduled job runs on a checkout of ``upstream/master``: it fails on an instruction class
that the registry does not know and on a constructor default that differs from the pinned one (rules R3, risks
7 and 8 of the plan). The registry and the option tables are loaded by path, so neither TensorFlow nor the
dependencies of the package are needed.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

ROOT_CLASS: Final = "tensorpotential.instructions.base.TPInstruction"
PACKAGE: Final = "tensorpotential"
SKIPPED_PACKAGES: Final = ("tensorpotential.compat",)
CAPTURE_DECORATOR: Final = "capture_init_args"
KIND_NAMES: Final = {
    "posonly": "POSITIONAL_ONLY",
    "regular": "POSITIONAL_OR_KEYWORD",
    "vararg": "VAR_POSITIONAL",
    "kwonly": "KEYWORD_ONLY",
    "kwarg": "VAR_KEYWORD",
}
_NO_DEFAULT: Final = object()
_REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Parameter:
    """One constructor parameter; ``default`` is the literal value or ``source`` text when it is not a literal."""

    name: str
    kind: str
    has_default: bool
    default: Any = None
    source: str | None = None

    def as_row(self) -> dict[str, str]:
        """The row of ``instruction_constructor_defaults.json`` (``default_repr`` is the repr of the default)."""
        row = {"name": self.name, "kind": self.kind}
        if self.has_default:
            row["default_repr"] = (
                self.source if self.source is not None else repr(self.default)
            )
        return row


@dataclass(frozen=True)
class ClassInfo:
    """An instruction class as the source states it."""

    path: str
    file: str
    line: int
    captured: bool
    parameters: tuple[Parameter, ...]

    def defaults(self) -> dict[str, Any]:
        """The literal defaults by parameter name (``name`` and the var-keyword parameter left out)."""
        return {
            p.name: p.default
            for p in self.parameters
            if p.has_default and p.source is None and p.name != "name"
        }

    def as_row(self) -> dict[str, Any]:
        """The row of the committed table."""
        return {
            "captured": self.captured,
            "parameters": [p.as_row() for p in self.parameters],
        }


@dataclass(frozen=True)
class _Raw:
    """A class definition before the bases are resolved."""

    path: str
    file: str
    node: ast.ClassDef
    bases: tuple[str, ...]


@dataclass(frozen=True)
class _Source:
    """Every class of the package and the names each module re-exports by import."""

    classes: dict[str, _Raw]
    aliases: dict[str, str]

    def resolve(self, path: str) -> str:
        """Follow ``from x import Name`` re-exports until a class definition is reached (else ``path``)."""
        seen: set[str] = set()
        while path not in self.classes and path in self.aliases and path not in seen:
            seen.add(path)
            path = self.aliases[path]
        return path


def module_name(root: Path, file: Path) -> str:
    """Dotted module name of ``file`` under ``root`` (``pkg/__init__.py`` is ``pkg``)."""
    parts = list(file.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(tree: ast.Module, module: str, is_package: bool) -> dict[str, str]:
    """Names this module binds by ``from x import y [as z]`` (relative imports resolved), as dotted paths."""
    bound: dict[str, str] = {}
    package = module if is_package else module.rpartition(".")[0]
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            for alias in node.names:
                bound[alias.asname or alias.name] = f"{base}.{alias.name}"
    return bound


def _base_path(
    expr: ast.expr, module: str, bound: Mapping[str, str], local: set[str]
) -> str:
    """Dotted path of a base-class expression, or ``""`` when it is not a name this module can resolve."""
    if isinstance(expr, ast.Name):
        if expr.id in local:
            return f"{module}.{expr.id}"
        return bound.get(expr.id, "")
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        head = bound.get(expr.value.id, "")
        return f"{head}.{expr.attr}" if head else ""
    return ""


def _source_files(root: Path) -> Iterator[Path]:
    for file in sorted((root / PACKAGE).rglob("*.py")):
        name = module_name(root, file)
        if not any(name == s or name.startswith(s + ".") for s in SKIPPED_PACKAGES):
            yield file


def _collect(root: Path) -> _Source:
    classes: dict[str, _Raw] = {}
    aliases: dict[str, str] = {}
    pending: list[tuple[str, str, ast.ClassDef, dict[str, str], set[str]]] = []
    for file in _source_files(root):
        module = module_name(root, file)
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        local = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        bound = _imports(tree, module, file.name == "__init__.py")
        aliases.update({f"{module}.{name}": target for name, target in bound.items()})
        pending += [
            (module, str(file.relative_to(root)), node, bound, local)
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        ]
    source = _Source(classes, aliases)
    for module, file, node, bound, local in pending:
        bases = tuple(
            source.resolve(b)
            for b in (_base_path(e, module, bound, local) for e in node.bases)
            if b
        )
        classes[f"{module}.{node.name}"] = _Raw(
            f"{module}.{node.name}", file, node, bases
        )
    return source


def _is_instruction(
    path: str, classes: Mapping[str, _Raw], seen: frozenset[str] = frozenset()
) -> bool:
    if path == ROOT_CLASS:
        return True
    raw = classes.get(path)
    if raw is None or path in seen:
        return False
    return any(_is_instruction(b, classes, seen | {path}) for b in raw.bases)


def _literal(node: ast.expr) -> Any:
    try:
        return ast.literal_eval(node)
    except ValueError:
        return _NO_DEFAULT


def _parameters(args: ast.arguments) -> tuple[Parameter, ...]:
    """The parameters of ``__init__`` without ``self``, in the order ``inspect.signature`` lists them."""
    found: list[Parameter] = []
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    for index, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = "posonly" if index < len(args.posonlyargs) else "regular"
        found.append(_parameter(arg.arg, KIND_NAMES[kind], default))
    if args.vararg:
        found.append(_parameter(args.vararg.arg, KIND_NAMES["vararg"], None))
    found += [
        _parameter(arg.arg, KIND_NAMES["kwonly"], default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    ]
    if args.kwarg:
        found.append(_parameter(args.kwarg.arg, KIND_NAMES["kwarg"], None))
    return tuple(p for p in found if p.name != "self")


def _parameter(name: str, kind: str, default: ast.expr | None) -> Parameter:
    if default is None:
        return Parameter(name, kind, has_default=False)
    value = _literal(default)
    if value is _NO_DEFAULT:
        return Parameter(name, kind, has_default=True, source=ast.unparse(default))
    return Parameter(name, kind, has_default=True, default=value)


def _own_init(node: ast.ClassDef) -> ast.FunctionDef | None:
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            return item
    return None


def _is_captured(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = (
            target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
        )
        if name == CAPTURE_DECORATOR:
            return True
    return False


def _constructor(
    path: str, classes: Mapping[str, _Raw], seen: frozenset[str] = frozenset()
) -> ast.FunctionDef | None:
    """The ``__init__`` that applies to ``path``: its own, else the first found in the bases, depth first."""
    raw = classes.get(path)
    if raw is None or path in seen:
        return None
    own = _own_init(raw.node)
    if own is not None:
        return own
    for base in raw.bases:
        found = _constructor(base, classes, seen | {path})
        if found is not None:
            return found
    return None


def extract(root: str | Path = _REPO) -> dict[str, ClassInfo]:
    """The instruction classes of the package under ``root``, by dotted path, sorted by path.

    Parameters
    ----------
    root
        A directory that contains ``tensorpotential/`` (a checkout, or a copy of one).
    """
    classes = _collect(Path(root)).classes
    table: dict[str, ClassInfo] = {}
    for path, raw in classes.items():
        if not _is_instruction(path, classes):
            continue
        init = _constructor(path, classes)
        table[path] = ClassInfo(
            path=path,
            file=raw.file,
            line=raw.node.lineno,
            captured=_is_captured(raw.node),
            parameters=_parameters(init.args) if init is not None else (),
        )
    return dict(sorted(table.items()))


def table_rows(table: Mapping[str, ClassInfo]) -> dict[str, dict[str, Any]]:
    """The table in the layout of ``tests/data/instruction_constructor_defaults.json``."""
    return {path: info.as_row() for path, info in table.items()}


def constructor_info(root: str | Path, path: str) -> ClassInfo | None:
    """The constructor of any class of the package (not only an instruction), or ``None`` when it is not defined."""
    classes = _collect(Path(root)).classes
    raw = classes.get(path)
    if raw is None:
        return None
    init = _constructor(path, classes)
    return ClassInfo(
        path=path,
        file=raw.file,
        line=raw.node.lineno,
        captured=_is_captured(raw.node),
        parameters=_parameters(init.args) if init is not None else (),
    )


def _import_registry() -> tuple[ModuleType, ModuleType]:
    """The registry and option modules of this checkout (importing the package does not import TensorFlow)."""
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    spec = "tensorpotential.torch_backend.spec"
    return (
        importlib.import_module(f"{spec}.registry"),
        importlib.import_module(f"{spec}.options"),
    )


def _plain(info: ClassInfo) -> set[str]:
    """Parameter names that a yaml can set: not ``name``, not ``*args`` or ``**kwargs``."""
    return {
        p.name
        for p in info.parameters
        if p.name != "name" and p.kind not in ("VAR_POSITIONAL", "VAR_KEYWORD")
    }


def completeness_problems(
    table: Mapping[str, ClassInfo], registry: ModuleType
) -> list[str]:
    """Classes the source defines and the registry lacks, and the other way round."""
    known = set(registry.REGISTRY)
    found = [
        f"{path} ({table[path].file}:{table[path].line}): instruction class without an entry in the registry; "
        "add it to SUPPORTED_OPTIONS or REJECTED_CLASSES in tensorpotential/torch_backend/spec/options.py"
        for path in sorted(table.keys() - known)
    ]
    found += [
        f"{path}: the registry names a class the source does not define (removed or renamed)"
        for path in sorted(known - table.keys())
    ]
    return found


def default_problems(
    root: str | Path,
    table: Mapping[str, ClassInfo],
    registry: ModuleType,
    options: ModuleType,
) -> list[str]:
    """What differs between the constructors of the supported classes and the option tables."""
    found: list[str] = []
    for cls in sorted(options.SUPPORTED_OPTIONS):
        info = table.get(cls)
        if info is None:
            continue  # reported by completeness_problems
        origin = options.DEFAULTS_ORIGIN.get(cls)
        source = constructor_info(root, origin) if origin else info
        if source is None:
            found.append(
                f"{cls}: the class {origin} that holds its pinned defaults is not defined any more"
            )
            continue
        found += _class_problems(cls, info, source, options)
    return found


def _class_problems(
    cls: str, info: ClassInfo, source: ClassInfo, options: ModuleType
) -> list[str]:
    short = cls.rsplit(".", 1)[-1]
    found: list[str] = []
    if not info.captured:
        found.append(
            f"{short}: capture_init_args was removed (the saved yaml would lose its defaults)"
        )
    rules = options.SUPPORTED_OPTIONS[cls]
    keys = _plain(info) | (_plain(source) if source is not info else set())
    found += [
        f"{short}: constructor parameter {key!r} has no option rule"
        for key in sorted(keys - rules.keys())
    ]
    found += [
        f"{short}: option {key!r} is no constructor parameter any more"
        for key in sorted(
            k for k, r in rules.items() if not r.ignored and k not in keys
        )
    ]
    pinned = options.DEFAULTS.get(cls, {})
    literal = source.defaults()
    for parameter in source.parameters:
        if parameter.has_default and parameter.source is not None:
            found.append(
                f"{short}.{parameter.name}: the default {parameter.source} is not a literal"
            )
    for key in sorted((set(pinned) | set(literal)) - {"name"}):
        # repr, so that 1, 1.0 and True (equal in Python) differ
        was = repr(literal[key]) if key in literal else "<absent>"
        now = repr(pinned[key]) if key in pinned else "<absent>"
        if was != now:
            found.append(f"{short}.{key}: source default {was}, pinned {now}")
    return found


def problems(
    root: str | Path, table: Mapping[str, ClassInfo] | None = None
) -> list[str]:
    """What differs between the source under ``root`` and the registry, one sentence each (empty when equal)."""
    registry, options = _import_registry()
    table = extract(root) if table is None else table
    return [
        *completeness_problems(table, registry),
        *default_problems(root, table, registry, options),
    ]


def main(argv: list[str] | None = None) -> int:
    """Print the table, or check it against the registry of this checkout."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("table", "check"))
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO,
        help="a checkout that holds tensorpotential/ (default: this repository)",
    )
    args = parser.parse_args(argv)
    table = extract(args.root)
    if args.command == "table":
        sys.stdout.write(json.dumps(table_rows(table), indent=1) + "\n")
        return 0
    found = problems(args.root, table)
    ok = f"ok: {len(table)} instruction classes, every one in the registry, pinned defaults equal the source"
    sys.stdout.write("\n".join(found or [ok]) + "\n")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
