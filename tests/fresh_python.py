"""Run Python code in a new interpreter that imports ``tensorpotential`` from this tree.

The one place that builds the environment of such a subprocess: ``PYTHONPATH`` starts with this
tree (the editable install may point at another checkout), TensorFlow is quiet and sees no GPU,
and nothing writes bytecode into the source tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 300


def run_fresh_python(
    code: str,
    cwd: Path,
    *,
    args: Sequence[str] = (),
    stdin: str | None = None,
    env: Mapping[str, str | None] | None = None,
    pythonpath: Sequence[Path | str] = (),
) -> subprocess.CompletedProcess:
    """Run ``code`` with ``python -W ignore -c``, passing ``args`` as ``sys.argv[1:]``.

    Parameters
    ----------
    code
        The program text.
    cwd
        Working directory of the child (a ``tmp_path``, never the source tree).
    args
        Command-line arguments after the code.
    stdin
        Text sent to the child's standard input.
    env
        Variables to set; a value of ``None`` removes the variable.
    pythonpath
        Directories searched before this tree.
    """
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None, [*map(str, pythonpath), str(REPO_ROOT), child_env.get("PYTHONPATH")]
        )
    )
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    child_env["CUDA_VISIBLE_DEVICES"] = "-1"
    for key, value in (env or {}).items():
        if value is None:
            child_env.pop(key, None)
        else:
            child_env[key] = value
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code, *args],
        input=stdin,
        cwd=cwd,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )


def last_stdout_line(
    code: str, cwd: Path, env: Mapping[str, str | None] | None = None
) -> str:
    """Run ``code`` like :func:`run_fresh_python`, require success and return the last line printed."""
    result = run_fresh_python(code, cwd, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]
