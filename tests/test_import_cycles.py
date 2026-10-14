"""No module of the package takes part in an import-time cycle.

An import at module level (also inside ``if``, ``try`` or a class body, not inside a function) runs when the module is
imported, so two modules that import each other that way depend on the order of the imports. Imports inside a function
run later and are not counted; they close larger cycles (listed on the ARCH1 issue) that the contracts of
``pyproject.toml`` do not forbid.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterator, Mapping
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "tensorpotential"
# legacy converter that needs the optional pyace package, out of scope
SKIPPED = {"compat"}


def _modules() -> dict[str, tuple[Path, bool]]:
    found: dict[str, tuple[Path, bool]] = {}
    for path in PACKAGE.rglob("*.py"):
        parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)
        if SKIPPED & set(parts):
            continue
        is_init = parts[-1] == "__init__"
        found[".".join(parts[:-1] if is_init else parts)] = (path, is_init)
    return found


def _import_time_nodes(node: ast.AST) -> Iterator[ast.Import | ast.ImportFrom]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Import | ast.ImportFrom):
            yield child
        if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            yield from _import_time_nodes(child)


def _targets(
    module: str,
    is_init: bool,
    node: ast.Import | ast.ImportFrom,
    known: Mapping[str, object],
) -> set[str]:
    """The modules of the package that ``node`` imports, the imported names that are submodules included."""
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    else:
        base = node.module or ""
        if node.level:
            package = module.split(".") if is_init else module.split(".")[:-1]
            package = package[: len(package) - (node.level - 1)]
            base = ".".join([*package, *([base] if base else [])])
        names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
    return {name for name in names if name in known and name != module}


def import_time_graph() -> dict[str, set[str]]:
    """Map each module of the package to the package modules it imports when it is imported."""
    modules = _modules()
    graph: dict[str, set[str]] = defaultdict(set)
    for module, (path, is_init) in modules.items():
        for node in _import_time_nodes(ast.parse(path.read_text())):
            graph[module] |= _targets(module, is_init, node, modules)
    return graph


def find_cycle(graph: Mapping[str, set[str]]) -> list[str]:
    """One cycle of ``graph`` as a list of modules (first and last equal), or an empty list."""
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as error:
        return list(error.args[1])
    return []


def test_no_module_of_the_package_is_in_an_import_time_cycle():
    assert find_cycle(import_time_graph()) == []


def test_find_cycle_reports_a_two_module_cycle_and_ignores_a_chain():
    assert set(find_cycle({"a": {"b"}, "b": {"a"}, "c": {"a"}})) == {"a", "b"}
    assert find_cycle({"a": {"b"}, "b": {"c"}}) == []


def test_import_time_nodes_skips_function_bodies_and_keeps_class_and_try_bodies():
    tree = ast.parse(
        "import a\ndef f():\n    import b\nclass C:\n    import c\ntry:\n    import d\nexcept ImportError:\n    pass\n"
    )
    names = sorted(
        alias.name
        for node in _import_time_nodes(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert names == ["a", "c", "d"]


def test_a_relative_import_of_a_sibling_is_resolved():
    known = {"pkg": 0, "pkg.a": 0, "pkg.b": 0}
    node = ast.parse("from . import b").body[0]
    assert isinstance(node, ast.ImportFrom)
    assert _targets("pkg.a", False, node, known) == {"pkg", "pkg.b"}
