"""A module of the library does not import a private name of a module in another package.

A name that starts with an underscore belongs to its module. Another package that needs it makes it public (the owner
renames it and may keep the old name as an alias) or moves the code. The imports below are the ones that exist today,
each with its reason; the list may only shrink. Imports between modules of one package (``uq/cli/build``) are internal to
that package and are not counted.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "tensorpotential"
SKIPPED_PARTS = {"compat"}  # legacy converter, out of scope for every issue
# Every name an importer may take from a module of another package, with the reason; "*" stands for every private name.
ALLOWED: dict[tuple[str, str, str], str] = {
    (
        "tensorpotential.cli.data",
        "tensorpotential.data.datasets",
        "*",
    ): "the old path re-exports every name of the moved module",
    (
        "tensorpotential.scripts._kokkos_export",
        "tensorpotential.export_kokkos",
        "*",
    ): "the old path re-exports every name of the moved module",
    (
        "tensorpotential.extra.gen_tensor.wizard",
        "tensorpotential.cli.prompts",
        "*",
    ): "the prompt helpers are private in cli/prompts; making them public is the owner's call (finding on ARCH2)",
}
# the TensorFlow options module is imported for its side effect, never for a name
IGNORED_NAMES = {"_tf_options"}


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _package_of(path: Path, module: str) -> str:
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def private_imports(root: Path = PACKAGE) -> Iterator[tuple[str, str, str]]:
    """Yield ``(importer, owner, name)`` for each ``from owner import _name`` of a module in another package."""
    for path in sorted(root.rglob("*.py")):
        if SKIPPED_PARTS & set(path.parts):
            continue
        importer = _module_name(path, root)
        package = _package_of(path, importer)
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if (
                not node.module.startswith(root.name)
                or node.module.rpartition(".")[0] == package
            ):
                continue
            for alias in node.names:
                if (
                    alias.name.startswith("_")
                    and not alias.name.startswith("__")
                    and alias.name not in IGNORED_NAMES
                ):
                    yield importer, node.module, alias.name


@pytest.fixture(scope="module")
def found() -> set[tuple[str, str, str]]:
    return set(private_imports())


def _allowed(site: tuple[str, str, str]) -> bool:
    importer, owner, name = site
    return (importer, owner, name) in ALLOWED or (importer, owner, "*") in ALLOWED


def test_no_library_module_imports_a_private_name_of_another_package(found):
    assert {site for site in found if not _allowed(site)} == set(), (
        "make the name public or move the code"
    )


def test_every_allowed_import_still_exists(found):
    stale = [
        key
        for key in ALLOWED
        if not any(site[:2] == key[:2] and key[2] in {"*", site[2]} for site in found)
    ]
    assert stale == [], "the import is gone: remove its entry"


def test_the_helpers_other_packages_import_are_public_and_the_old_names_resolve_to_the_same_objects():
    from tensorpotential.uq import factories, gmmuq

    assert (
        factories.basis_rp_spec_from_artifact is factories._basis_rp_spec_from_artifact
    )
    assert gmmuq.MISSING_ELEM_INV_COV_DIAG == gmmuq._MISSING_ELEM_INV_COV_DIAG == 1e6


def test_a_planted_private_import_is_found(tmp_path):
    root = tmp_path / "pkg"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    for init in (
        root / "__init__.py",
        root / "a" / "__init__.py",
        root / "b" / "__init__.py",
    ):
        init.write_text("")
    (root / "b" / "owner.py").write_text("def _helper(): ...\n")
    (root / "a" / "user.py").write_text("from pkg.b.owner import _helper\n")
    assert list(private_imports(root)) == [("pkg.a.user", "pkg.b.owner", "_helper")]


def test_an_import_inside_one_package_and_a_public_import_are_not_found(tmp_path):
    root = tmp_path / "pkg"
    (root / "a").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "a" / "__init__.py").write_text("from pkg.a.one import _internal\n")
    (root / "a" / "one.py").write_text("def _internal(): ...\ndef public(): ...\n")
    (root / "a" / "two.py").write_text(
        "from pkg.a.one import _internal\nfrom pkg import a\nfrom pkg.a.one import public\n"
    )
    assert list(private_imports(root)) == []
