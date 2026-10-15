"""The capability contract of ``tests_torch`` (``capabilities.py`` and the hooks of ``conftest.py``).

Logic of the helpers in-process; the hooks end to end, by running pytest in a fresh interpreter on a directory that
holds a copy of the real ``conftest.py`` and a planted test file. The ``hpc`` capability is the one whose availability
a test controls (``GRACE_HPC``), so the end-to-end runs use it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.fresh_python import run_fresh_python
from tests_torch import capabilities as caps

CONFTEST = Path(__file__).resolve().parent / "conftest.py"
RUN_PYTEST = "import sys, pytest; sys.exit(pytest.main([*sys.argv[1:], '-p', 'no:cacheprovider', '-q']))"
USAGE_ERROR = 4
NO_TESTS_COLLECTED = 5

HPC_TEST = """
import pytest

@pytest.mark.hpc
def test_on_the_hpc():
    pass

def test_anywhere():
    pass
"""


def _run(
    tmp_path: Path, test_text: str, env: dict[str, str | None], *args: str
) -> tuple[int, str]:
    (tmp_path / "conftest.py").write_text(CONFTEST.read_text())
    (tmp_path / "test_planted.py").write_text(test_text)
    done = run_fresh_python(
        RUN_PYTEST, tmp_path, args=[str(tmp_path), "-rs", *args], env=env
    )
    return done.returncode, done.stdout + done.stderr


@pytest.fixture
def no_caps_env() -> dict[str, str | None]:
    return {caps.REQUIRE_VARIABLE: None, caps.HPC_VARIABLE: None}


# ------------------------------------------------------------------ logic: helpers


def test_required_reads_a_comma_separated_list():
    assert caps.required({caps.REQUIRE_VARIABLE: " vesin, torchsim ,,"}) == {
        "vesin",
        "torchsim",
    }
    assert caps.required({}) == frozenset()


def test_required_rejects_an_unknown_name_and_lists_the_known_ones():
    with pytest.raises(
        ValueError,
        match=r"unknown capabilities torch_sim; the known ones are tf, torchsim",
    ):
        caps.required({caps.REQUIRE_VARIABLE: "vesin,torch_sim"})


def test_missing_keeps_unavailable_capabilities_and_ignores_other_markers():
    available = {"tf": False, "vesin": True, "hpc": False}.__getitem__
    assert caps.missing(["hpc", "parametrize", "vesin", "tf", "hpc"], available) == [
        "hpc",
        "tf",
    ]


def test_the_skip_reason_says_what_is_needed():
    reason = caps.skip_reason(["vesin", "hpc"])
    assert reason.startswith("needs hpc: the job runs on the HPC")
    assert "; vesin: the vesin neighbour list" in reason


def test_the_empty_parameter_set_reason_is_recognised():
    assert caps.is_empty_parameter_set("got empty parameter set for (backend)")
    assert not caps.is_empty_parameter_set(
        "needs vesin: the vesin neighbour list is installed"
    )


def test_the_hpc_probe_reads_the_job_declaration(monkeypatch):
    probe = caps.CAPABILITIES["hpc"].probe
    monkeypatch.setenv(caps.HPC_VARIABLE, "1")
    assert probe()
    monkeypatch.setenv(caps.HPC_VARIABLE, "0")
    assert not probe()


def test_has_package_looks_a_package_up_without_importing_it():
    assert "tabnanny" not in sys.modules
    assert caps.has_package("tabnanny")
    assert "tabnanny" not in sys.modules
    assert not caps.has_package("no_such_package_of_grace")


def test_has_package_answers_false_when_the_lookup_fails(monkeypatch):
    def refuse(name):
        raise ImportError(name)

    monkeypatch.setattr(caps.importlib.util, "find_spec", refuse)
    assert not caps.has_package("vesin")


def test_the_gpu_probe_answers_a_bool():
    assert isinstance(caps.CAPABILITIES["gpu"].probe(), bool)


def test_every_capability_and_marker_says_what_it_means():
    assert all(c.description for c in caps.CAPABILITIES.values())
    assert all(caps.OTHER_MARKERS.values())


# ------------------------------------------------------------------ logic: the hooks, end to end


def test_an_unavailable_capability_skips_with_the_reason(tmp_path, no_caps_env):
    code, out = _run(tmp_path, HPC_TEST, no_caps_env)
    assert code == 0, out
    assert "1 passed, 1 skipped" in out
    assert "needs hpc: the job runs on the HPC" in out


def test_an_available_capability_runs_the_test(tmp_path, no_caps_env):
    code, out = _run(tmp_path, HPC_TEST, {**no_caps_env, caps.HPC_VARIABLE: "1"})
    assert code == 0, out
    assert "2 passed" in out
    assert "skipped" not in out


def test_a_guaranteed_capability_that_is_missing_stops_the_run(tmp_path, no_caps_env):
    code, out = _run(tmp_path, HPC_TEST, {**no_caps_env, caps.REQUIRE_VARIABLE: "hpc"})
    assert code == USAGE_ERROR, out
    assert "GRACE_REQUIRE_CAPS guarantees hpc, which this environment lacks" in out


def test_a_guaranteed_capability_that_is_present_runs_everything(tmp_path, no_caps_env):
    env = {caps.REQUIRE_VARIABLE: "hpc", caps.HPC_VARIABLE: "1"}
    code, out = _run(tmp_path, HPC_TEST, {**no_caps_env, **env})
    assert code == 0, out
    assert "2 passed" in out


def test_an_unknown_guaranteed_capability_stops_the_run(tmp_path, no_caps_env):
    code, out = _run(tmp_path, HPC_TEST, {**no_caps_env, caps.REQUIRE_VARIABLE: "vesn"})
    assert code == USAGE_ERROR, out
    assert "names unknown capabilities vesn" in out


def test_a_parametrisation_over_an_empty_list_fails_the_collection(
    tmp_path, no_caps_env
):
    text = (
        "import pytest\n\n"
        "@pytest.mark.parametrize('backend', [])\n"
        "def test_every_backend(backend):\n    pass\n\n"
        "def test_other():\n    pass\n"
    )
    code, out = _run(tmp_path, text, no_caps_env)
    assert code == USAGE_ERROR, out
    assert (
        "parametrised over an empty list (nothing would run): test_planted.py::test_every_backend"
        in out
    )


def test_a_run_that_collects_nothing_fails(tmp_path, no_caps_env):
    # pytest's own exit code, kept as a test because the issue's zero-collection guard and the CI job rely on it
    code, out = _run(tmp_path, "def helper():\n    pass\n", no_caps_env)
    assert code == NO_TESTS_COLLECTED, out


def test_every_marker_is_registered_for_a_strict_run(tmp_path, no_caps_env):
    marks = "".join(
        f"@pytest.mark.{name}\n" for name in [*caps.CAPABILITIES, *caps.OTHER_MARKERS]
    )
    code, out = _run(
        tmp_path,
        f"import pytest\n\n{marks}def test_marked():\n    pass\n",
        no_caps_env,
        "--strict-markers",
    )
    assert code == 0, out
    assert "1 skipped" in out
