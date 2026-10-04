"""Run Python code in a new interpreter that imports ``tensorpotential`` from this tree."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 300


def run_fresh_python(
    code: str, tmp_path: Path, **env_overrides: str | None
) -> subprocess.CompletedProcess:
    """Run ``code`` in a new interpreter that imports ``tensorpotential`` from this tree.

    A value of ``None`` in ``env_overrides`` removes the variable.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")])
    )
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
