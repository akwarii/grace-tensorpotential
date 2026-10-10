"""Tests for ``tensorpotential.core.lazy`` (``lazy_exports`` and the TensorFlow ImportError).

A small package written into ``tmp_path`` stands in for ``tensorpotential``: its ``__init__`` calls
``lazy_exports`` exactly as the real ones do, one of its modules is fine, one needs a missing
third-party module and one needs ``keras``.

Logic layer: the module is imported on first access and not before, the object is cached in the
package, unknown names raise ``AttributeError``, ``dir`` lists the names without importing them,
an ``ImportError`` that is not about TensorFlow passes through unchanged. Behaviour layer
(fresh interpreter with a finder that refuses ``tensorflow`` and ``keras``): the error names the
requested object and the ``tf`` extra and keeps the original error as its cause.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from tensorpotential.core.backends import BackendNotInstalledError
from tensorpotential.core.lazy import is_tf_import_error, lazy_exports, missing_tf_error
from tests.fresh_python import run_fresh_python

PACKAGE = "lazy_demo_pkg"
FILES = {
    "__init__.py": f"""
        from tensorpotential.core.lazy import lazy_exports

        __getattr__, __dir__ = lazy_exports(
            __name__,
            {{
                "Thing": "{PACKAGE}.fine",
                "NeedsTf": "{PACKAGE}.needs_tf",
                "NeedsOther": "{PACKAGE}.needs_other",
            }},
        )
    """,
    "fine.py": "class Thing:\n    pass\n",
    "needs_tf.py": "import keras\n\nclass NeedsTf:\n    pass\n",
    "needs_other.py": "import no_such_module_for_tensorpotential_tests\n\nclass NeedsOther:\n    pass\n",
}


@pytest.fixture
def demo_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / PACKAGE
    root.mkdir()
    for name, text in FILES.items():
        (root / name).write_text(textwrap.dedent(text))
    monkeypatch.syspath_prepend(str(tmp_path))
    for module in [
        m for m in sys.modules if m == PACKAGE or m.startswith(PACKAGE + ".")
    ]:
        monkeypatch.delitem(sys.modules, module)
    return tmp_path


def test_first_access_imports_the_module_and_later_accesses_use_the_cache(
    demo_package: Path,
) -> None:
    import lazy_demo_pkg

    assert f"{PACKAGE}.fine" not in sys.modules
    thing = lazy_demo_pkg.Thing
    assert f"{PACKAGE}.fine" in sys.modules
    assert thing is sys.modules[f"{PACKAGE}.fine"].Thing
    assert vars(lazy_demo_pkg)["Thing"] is thing
    assert lazy_demo_pkg.Thing is thing


def test_from_import_goes_through_the_lazy_getattr(demo_package: Path) -> None:
    from lazy_demo_pkg import Thing

    assert Thing.__module__ == f"{PACKAGE}.fine"


def test_unknown_name_is_an_attribute_error(demo_package: Path) -> None:
    import lazy_demo_pkg

    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        lazy_demo_pkg.Missing  # noqa: B018
    assert not hasattr(lazy_demo_pkg, "Missing")


def test_dir_lists_the_lazy_names_without_importing_them(demo_package: Path) -> None:
    import lazy_demo_pkg

    names = dir(lazy_demo_pkg)
    assert {"Thing", "NeedsTf", "NeedsOther"} <= set(names)
    assert names == sorted(names)
    assert f"{PACKAGE}.fine" not in sys.modules


def test_an_import_error_that_is_not_about_tensorflow_passes_through(
    demo_package: Path,
) -> None:
    import lazy_demo_pkg

    with pytest.raises(
        ModuleNotFoundError, match="no_such_module_for_tensorpotential_tests"
    ) as info:
        lazy_demo_pkg.NeedsOther  # noqa: B018
    assert info.value.__cause__ is None


BLOCKED_TF = """
import sys

class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("tensorflow", "keras"):
            raise ImportError(f"No module named {name!r}", name=name)

sys.meta_path.insert(0, Refuse())
import lazy_demo_pkg

try:
    lazy_demo_pkg.NeedsTf
except ImportError as exc:
    print("MESSAGE", exc)
    print("NAME", exc.name)
    print("CAUSE", type(exc.__cause__).__name__, exc.__cause__)
"""


def test_a_missing_tensorflow_is_reported_with_the_name_and_the_extra(
    demo_package: Path,
) -> None:
    result = run_fresh_python(BLOCKED_TF, demo_package, pythonpath=[demo_package])
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "'NeedsTf' needs TensorFlow" in out
    assert "tensorpotential[tf]" in out
    assert "NAME keras" in out
    assert "CAUSE ImportError No module named 'keras'" in out


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tensorflow", True),
        ("tensorflow.python.framework", True),
        ("keras", True),
        ("keras.src", True),
        ("tensorflowx", False),
        ("numpy", False),
        ("tensorpotential.tpmodel", False),
        (None, False),
    ],
)
def test_is_tf_import_error(name: str | None, expected: bool) -> None:
    assert is_tf_import_error(ImportError("x", name=name)) is expected


def test_missing_tf_error_keeps_the_original_text_and_name() -> None:
    original = ImportError("No module named 'tensorflow'", name="tensorflow")
    error = missing_tf_error("TPModel", original)
    assert "No module named 'tensorflow'" in str(error)
    assert "'TPModel'" in str(error)
    assert error.name == "tensorflow"


def test_lazy_exports_returns_two_functions() -> None:
    getattr_, dir_ = lazy_exports("tensorpotential", {})
    assert callable(getattr_)
    assert callable(dir_)


def test_missing_tf_error_does_not_wrap_an_error_that_already_carries_the_hint() -> (
    None
):
    original = BackendNotInstalledError("tf", ("tensorflow", "keras"), "text")
    error = missing_tf_error("TPCalculator", original)
    assert isinstance(error, BackendNotInstalledError)
    assert (error.backend, error.missing) == ("tf", ("tensorflow", "keras"))
    assert str(error).startswith(
        "'TPCalculator' needs TensorFlow, which is not installed"
    )
    assert "missing: tensorflow, keras" in str(error)
    assert "could not be imported" not in str(error)
    assert "pip install 'tensorpotential[tf]'" in str(error)
