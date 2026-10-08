"""Smoke test of an installed ``tensorpotential`` wheel, run with the Python of a clean virtual environment.

CI installs the built wheel (with an extra, or none) into a fresh venv and runs::

    python tools/wheel_smoke.py VENV_PYTHON [--expect tf|no-tf]

The checks need nothing but the standard library, so the tool runs under any Python:

1. the TensorFlow-free core (``tensorpotential.core`` and ``tensorpotential.functions.couplings``)
   imports, and importing it does not load TensorFlow, even when TensorFlow is installed;
2. ``--help`` of every console script that the wheel registers (read from its ``entry_points.txt``):
   - with TensorFlow installed, the usage text and exit status 0;
   - without TensorFlow, the scripts that need it exit non-zero and name ``pip install 'tensorpotential[tf]'``,
     the others print their usage text.

``--expect`` fails the run when the environment is not the one the job meant to test (a torch-only job
must not silently have TensorFlow). ``grace_dashboard`` needs ``flask``, which no extra installs (finding of
CLEAN3); it is expected to name the missing module until flask is installed or becomes a dependency.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

CORE_IMPORTS = """
import sys
import tensorpotential
import tensorpotential.core.backends
import tensorpotential.core.cutoffs
import tensorpotential.core.lazy
import tensorpotential.functions.couplings
loaded = sorted(m for m in ("tensorflow", "tf_keras") if m in sys.modules)
print("LOADED", ",".join(loaded))
"""

PROBE = """
import importlib.util
print(",".join(m for m in ("tensorflow", "tf_keras", "torch", "flask") if importlib.util.find_spec(m)))
"""

#: Console scripts that cannot run without TensorFlow (no ``--help`` either): found by running them in an
#: environment without it (``tests/test_tf_options.py`` pins the four of the plan; gracemaker needs it as well).
NEED_TF = frozenset({
    "gracemaker",
    "extxyz2df",
    "grace_preprocess",
    "grace_predict",
    "grace_utils",
})
NEED_FLASK = frozenset({"grace_dashboard"})
HINT = "pip install 'tensorpotential[tf]'"
TIMEOUT_S = 600


def run(python: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python`` with ``args`` in a scratch environment that hides the repository from ``sys.path``.

    ``PYTHONPATH`` is passed on when set: a CI venv has none, a worktree checkout needs it.
    """
    env = {
        "PATH": str(Path(python).parent),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return subprocess.run(
        [str(python), *args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
        cwd=Path(python).parent,
        env=env,
    )


def installed_packages(python: Path) -> set[str]:
    """The names of tensorflow, tf_keras, torch and flask that this environment can import."""
    result = run(python, "-c", PROBE)
    return set(filter(None, result.stdout.strip().split(",")))


def console_scripts(python: Path) -> list[str]:
    """The console scripts that the installed wheel registers."""
    code = (
        "import importlib.metadata as md\n"
        "print(*sorted(e.name for e in md.distribution('tensorpotential').entry_points "
        "if e.group == 'console_scripts'))"
    )
    return run(python, "-c", code).stdout.split()


def check_core(python: Path) -> list[str]:
    """Problems with importing the TensorFlow-free core, as messages."""
    result = run(python, "-c", CORE_IMPORTS)
    if result.returncode != 0:
        return [f"the TensorFlow-free core does not import:\n{result.stderr.strip()}"]
    loaded = result.stdout.strip().rsplit("LOADED", 1)[-1].strip()
    return [f"importing the core loaded {loaded}"] if loaded else []


def check_script(python: Path, name: str, have: set[str]) -> str | None:
    """Run ``<name> --help`` of the environment; return the problem, or None when it behaves as expected."""
    exe = Path(python).parent / name
    result = run(exe, "--help")
    out, err = result.stdout, result.stderr
    if name in NEED_FLASK and "flask" not in have:
        return (
            None
            if "flask" in err and result.returncode != 0
            else f"{name}: expected a missing-flask error, got rc={result.returncode}"
        )
    if name in NEED_TF and "tensorflow" not in have:
        if result.returncode == 0 or HINT not in err:
            return f"{name}: expected a non-zero exit and {HINT!r}, got rc={result.returncode}: {err.strip()[-300:]}"
        return None
    # grace_uq prints its own help, which starts with the command name instead of "usage:"
    if result.returncode != 0 or not (name in out or "usage" in out.lower()):
        return f"{name}: expected the usage text and exit 0, got rc={result.returncode}: {err.strip()[-300:]}"
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the smoke test; return the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "python",
        type=Path,
        help="python of the clean venv that has the wheel installed",
    )
    parser.add_argument(
        "--expect",
        choices=["tf", "no-tf"],
        help="fail unless TensorFlow is (not) installed there",
    )
    args = parser.parse_args(argv)

    # not resolve(): a venv python is a symlink to the base one
    args.python = Path(os.path.abspath(args.python))
    have = installed_packages(args.python)
    print(
        f"environment has: {sorted(have) or 'none of tensorflow, tf_keras, torch, flask'}"
    )
    problems: list[str] = []
    if args.expect and (("tensorflow" in have) != (args.expect == "tf")):
        problems.append(
            f"--expect {args.expect}, but the environment has {sorted(have)}"
        )
    problems += check_core(args.python)
    scripts = console_scripts(args.python)
    if len(scripts) != 10:
        problems.append(f"expected 10 console scripts, found {len(scripts)}: {scripts}")
    for name in scripts:
        problem = check_script(args.python, name, have)
        print(f"{'ok  ' if problem is None else 'FAIL'} {name} --help")
        if problem:
            problems.append(problem)
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
