"""CPU benchmark of start-up and of the hot paths of the TensorFlow side (QUAL9).

Every measurement runs in a fresh interpreter (``python tools/bench_cpu.py case NAME``) that prints one
JSON line (peak resident memory and the phases it timed); ``run`` repeats each case, checks that the
machine is idle before, between and after, and prints a markdown table of median and spread. The table
depends on the machine: it goes into the pull request and the issue, never into ``baselines/``.

Cases: ``import:MODULE`` and ``script:NAME`` (the ``--help`` of a console script of ``pyproject.toml``),
``calc_init:YAML`` (jit-decorated, as the calculator tests do), ``builder:small|medium|large`` (neighbour list per structure), ``train_step`` and
``train_run`` (tiny model), ``calc_eval`` (``TPCalculator`` loaded from the SavedModel of the tests, 100 structures).

Usage (from the repository root, machine idle)::

    python tools/bench_cpu.py run --out bench.json
    python tools/bench_cpu.py run --cases import:tensorflow train_step --repeats 5
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
YAMLS = (
    "model_grace.yaml",
    "model_grace_2L_omat.yaml",
    "model_grace_2L_omat_large_base.yaml",
)
SAVED_MODEL = (
    TESTS / "test_calculator_model"
)  # a SavedModel on Mo, Nb, Ta, W, the way users load one
STRUCTURES = TESTS / "data" / "MoNbTaW_test50.pkl.gz"
BUILDER_SIZES = {
    "small": 1,
    "medium": 2,
    "large": 4,
}  # supercell edge of the 54-atom cell
TRAIN_PRESET = "GRACE_2LAYER_v1_24"
MAX_LOAD = 4.0  # CLAUDE.md: busy above this one-minute load
MAX_FOREIGN_CORES = (
    0.5  # other processes' cores during a run above which a sample is marked
)
BUSY_PATTERN = re.compile(r"[p]ytest|[g]racemaker")


def console_scripts() -> dict[str, str]:
    """Return the console scripts of ``pyproject.toml`` as ``name -> module:function``."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def default_cases() -> list[str]:
    """Every case of the issue, in table order."""
    cases = [
        "import:tensorflow",
        "import:tensorpotential",
        "import:tensorpotential.calculator",
    ]
    cases += [f"script:{name}" for name in console_scripts()]
    cases += [f"calc_init:{name}" for name in YAMLS]
    cases += [f"builder:{size}" for size in BUILDER_SIZES]
    return [*cases, "train_step", "train_run", "calc_eval"]


# ------------------------------------------------------------------ cases (run in the child)


def _timed(phases: dict[str, float], name: str, fn: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    result = fn()
    phases[name] = phases.get(name, 0.0) + (time.perf_counter() - start) * 1e3
    return result


def _atoms_list() -> list:
    import pandas as pd

    return list(pd.read_pickle(STRUCTURES)["ase_atoms"])  # noqa: S301 - repo test data


def _import_case(spec: str) -> dict[str, float]:
    import importlib

    if spec == "tensorflow":
        phases: dict[str, float] = {}
        _timed(phases, "import_ms", lambda: importlib.import_module("tensorflow"))
        return phases
    return {"import_ms": _import_time(spec)}


def _import_time(module: str) -> float:
    import importlib

    start = time.perf_counter()
    importlib.import_module(module)
    return (time.perf_counter() - start) * 1e3


def _script_case(name: str) -> dict[str, float]:
    import importlib

    module, _, function = console_scripts()[name].partition(":")
    phases: dict[str, float] = {}
    main = _timed(
        phases, "import_ms", lambda: getattr(importlib.import_module(module), function)
    )
    sys.argv = [name, "--help"]
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.suppress(SystemExit):
        main()
    phases["help_ms"] = (time.perf_counter() - start) * 1e3
    return phases


def _tensorflow_phases(phases: dict[str, float]) -> None:
    import importlib

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    _timed(
        phases,
        "import_tensorpotential_ms",
        lambda: importlib.import_module("tensorpotential"),
    )
    _timed(
        phases, "import_tensorflow_ms", lambda: importlib.import_module("tensorflow")
    )


def _model_of(yaml_name: str):
    sys.path.insert(0, str(ROOT / "tools"))
    import grace_probe

    return grace_probe.build_probe_model(grace_probe.resolve_yaml(yaml_name), "float64")


def _calc_init_case(yaml_name: str) -> dict[str, float]:
    phases: dict[str, float] = {}
    _tensorflow_phases(phases)
    from tensorpotential.calculator import TPCalculator

    model = _timed(phases, "build_model_ms", lambda: _model_of(yaml_name))
    # as the calculator tests and the foundation-model loader do: an undecorated model runs eagerly
    _timed(
        phases, "decorate_ms", lambda: model.decorate_compute_function(jit_compile=True)
    )
    _timed(phases, "construct_calculator_ms", lambda: TPCalculator(model))
    return phases


def _builder_case(size: str, repeats: int = 10) -> dict[str, float]:
    phases: dict[str, float] = {}
    _tensorflow_phases(phases)
    from tensorpotential.data.databuilder import GeometricalDataBuilder

    atoms = max(_atoms_list(), key=len).repeat((BUILDER_SIZES[size],) * 3)
    builder = GeometricalDataBuilder({"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, cutoff=6.0)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        builder.join_to_batch([builder.extract_from_ase_atoms(atoms.copy())])
        times.append((time.perf_counter() - start) * 1e3)
    return {"n_atoms": float(len(atoms)), "neighbour_list_ms": statistics.median(times)}


def _reference_atoms(count: int) -> list:
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator

    rng = np.random.default_rng(0)
    structures = []
    for original in _atoms_list()[:count]:
        atoms = original.copy()
        atoms.calc = SinglePointCalculator(
            atoms, energy=float(rng.normal()), forces=rng.normal(size=(len(atoms), 3))
        )
        structures.append(atoms)
    return structures


def _tiny_potential():
    from tensorpotential.loss import (
        LossFunction,
        WeightedSSEEnergyLoss,
        WeightedSSEForceLoss,
    )
    from tensorpotential.metrics import ComputeMetrics, EnergyMetrics, ForceMetrics
    from tensorpotential.potentials import get_preset
    from tensorpotential.tensorpot import TensorPotential

    elements = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
    instructions = get_preset(TRAIN_PRESET)(
        element_map=elements, lmax=0
    ).get_instructions()
    return TensorPotential(
        instructions,
        loss_function=LossFunction({
            "energy": WeightedSSEEnergyLoss(1.0),
            "forces": WeightedSSEForceLoss(1.0),
        }),
        compute_metrics=ComputeMetrics([EnergyMetrics(), ForceMetrics()]),
        global_batch_size=8,
        fit_config={"optimizer": "Adam", "opt_params": {"learning_rate": 1e-3}},
        jit_compile=True,
    )


def _train_case(steps: int, structures: int = 24) -> dict[str, float]:
    phases: dict[str, float] = {}
    _tensorflow_phases(phases)
    import tensorflow as tf

    from tensorpotential.data.databuilder import (
        AbstractDataBuilder,
        GeometricalDataBuilder,
        ReferenceEnergyForcesStressesDataBuilder,
        construct_batches,
    )

    tp = _timed(phases, "build_model_ms", _tiny_potential)
    builders: list[AbstractDataBuilder] = [
        GeometricalDataBuilder({"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}, cutoff=5.0),
        ReferenceEnergyForcesStressesDataBuilder(),
    ]
    specs = tp.get_model_grad_signatures()
    batches = _timed(
        phases,
        "dataset_ms",
        lambda: [
            {
                key: tf.constant(batch[key], dtype=spec.dtype)
                for key, spec in specs.items()
            }
            for batch in construct_batches(
                _reference_atoms(structures), builders, 8, verbose=False
            )
        ],
    )
    _timed(phases, "first_step_ms", lambda: tp.distributed_train_step(batches[0]))
    for step in range(steps - 1):
        _timed(
            phases,
            "steady_steps_ms",
            lambda step=step: tp.distributed_train_step(batches[step % len(batches)]),
        )
    return phases


def _calc_eval_case(structures: int = 100) -> dict[str, float]:
    phases: dict[str, float] = {}
    _tensorflow_phases(phases)
    from tensorpotential.calculator import TPCalculator

    calc = _timed(
        phases, "load_calculator_ms", lambda: TPCalculator(model=str(SAVED_MODEL))
    )
    for compiled in calc.models:
        compiled.compute = _wrapped(phases, "model_ms", compiled.compute)
    calc.get_data = _wrapped(phases, "data_ms", calc.get_data)
    atoms_list = _atoms_list()
    start = time.perf_counter()
    for index in range(structures):
        atoms = atoms_list[index % len(atoms_list)].copy()
        atoms.calc = calc
        atoms.get_potential_energy()
        atoms.get_forces()
    phases["evaluate_total_ms"] = (time.perf_counter() - start) * 1e3
    phases["other_ms"] = (
        phases["evaluate_total_ms"] - phases["model_ms"] - phases["data_ms"]
    )
    return phases


def _wrapped(
    phases: dict[str, float], name: str, fn: Callable[..., Any]
) -> Callable[..., Any]:
    def timed(*args: Any, **kwargs: Any) -> Any:
        return _timed(phases, name, lambda: fn(*args, **kwargs))

    return timed


def peak_rss_mb(status: Path = Path("/proc/self/status")) -> float:
    """Peak resident memory of this process in MB.

    ``VmHWM`` restarts at ``exec``; ``ru_maxrss`` would carry the peak of a large parent through ``fork``.
    """
    for line in status.read_text().splitlines() if status.exists() else []:
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_case(name: str, steps: int = 100, structures: int = 100) -> dict[str, Any]:
    """Run one case in this process; its phases in ms and the peak resident memory in MB."""
    kind, _, argument = name.partition(":")
    runners: dict[str, Callable[[], dict[str, float]]] = {
        "import": lambda: _import_case(argument),
        "script": lambda: _script_case(argument),
        "calc_init": lambda: _calc_init_case(argument),
        "builder": lambda: _builder_case(argument),
        "train_step": lambda: _train_case(3),
        "train_run": lambda: _train_case(steps),
        "calc_eval": lambda: _calc_eval_case(structures),
    }
    if kind not in runners:
        raise SystemExit(  # noqa: TRY003
            f"unknown case {name!r}; known: {', '.join(default_cases())}"
        )
    phases = runners[kind]()
    return {"phases_ms": phases, "peak_rss_mb": peak_rss_mb()}


# ------------------------------------------------------------------ driver


def busy_processes(ps_output: str) -> list[str]:
    """Return the lines of ``ps -eo pid,etimes,args`` that are test or training processes."""
    return [line for line in ps_output.splitlines() if BUSY_PATTERN.search(line)]


def idle_problem(check_load: bool = True) -> str | None:
    """Why the machine is not idle (one-minute load above ``MAX_LOAD`` if ``check_load``, a test or training process), or ``None``.

    The benchmark's own interpreters raise the load average, so only the first check reads it.
    """
    load = os.getloadavg()[0]
    ps = subprocess.run(
        ["ps", "-eo", "pid,etimes,args"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    others = [
        line for line in busy_processes(ps) if f" {os.getpid()} " not in f" {line} "
    ]
    if check_load and load > MAX_LOAD:
        return f"one-minute load {load:.2f} is above {MAX_LOAD}"
    return f"{len(others)} test or training process(es) running" if others else None


def run_sample(name: str, steps: int, structures: int) -> dict[str, Any]:
    """One fresh interpreter running ``name``: wall time, other processes' cores, and what the child printed."""
    sys.path.insert(0, str(ROOT / "tools"))
    import grace_probe

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "case",
        name,
        "--steps",
        str(steps),
        "--structures",
        str(structures),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }
    machine0 = grace_probe.machine_busy_seconds()
    children0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    proc = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, check=False, env=env, cwd=ROOT
    )
    wall = time.perf_counter() - start
    children1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    own = (
        children1.ru_utime
        + children1.ru_stime
        - children0.ru_utime
        - children0.ru_stime
    )
    foreign = grace_probe.foreign_cores(
        machine0, grace_probe.machine_busy_seconds(), own, wall
    )
    sample: dict[str, Any] = {"wall_ms": wall * 1e3, "foreign_cores": foreign}
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        return {**sample, "error": lines[-1] if lines else f"exit {proc.returncode}"}
    return {**sample, **json.loads(proc.stdout.strip().splitlines()[-1])}


def _spread(values: Sequence[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summarise(name: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Median and spread over the samples that ran; failures are counted and the last error kept."""
    good = [s for s in samples if "error" not in s]
    row: dict[str, Any] = {
        "case": name,
        "runs": len(samples),
        "failed": len(samples) - len(good),
    }
    if not good:
        return {**row, "error": samples[-1]["error"]}
    row["wall_ms"] = _spread([s["wall_ms"] for s in good])
    row["peak_rss_mb"] = _spread([s["peak_rss_mb"] for s in good])
    row["phases_ms"] = {
        key: statistics.median(s["phases_ms"][key] for s in good)
        for key in good[0]["phases_ms"]
    }
    foreign = [s["foreign_cores"] for s in good if s["foreign_cores"] is not None]
    row["foreign_cores_max"] = max(foreign) if foreign else None
    return row


def render_table(rows: list[dict[str, Any]]) -> str:
    """Return the markdown table of the summary rows."""
    lines = [
        "| case | runs | wall ms: median [min, max] | peak RSS MB | phases (median ms) | other cores (max) |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if "error" in row:
            lines.append(
                f"| {row['case']} | {row['runs']} | failed: {row['error']} | | | |"
            )
            continue
        wall, rss = row["wall_ms"], row["peak_rss_mb"]
        phases = ", ".join(
            f"{key.removesuffix('_ms')} {value:.1f}"
            for key, value in row["phases_ms"].items()
        )
        foreign = (
            "n/a"
            if row["foreign_cores_max"] is None
            else f"{row['foreign_cores_max']:.2f}"
        )
        lines.append(
            f"| {row['case']} | {row['runs']} | {wall['median']:.0f} [{wall['min']:.0f}, {wall['max']:.0f}]"
            f" | {rss['median']:.0f} | {phases} | {foreign} |"
        )
    return "\n".join(lines)


def wait_until_idle(
    check_load: bool, wait_s: float, poll_s: float = 15.0
) -> str | None:
    """Poll ``idle_problem`` for up to ``wait_s`` seconds; return the last problem, ``None`` once idle."""
    deadline = time.monotonic() + wait_s
    problem = idle_problem(check_load=check_load)
    while problem and time.monotonic() < deadline:
        time.sleep(poll_s)
        problem = idle_problem(check_load=check_load)
    return problem


def run(
    cases: Sequence[str],
    repeats: int,
    steps: int,
    structures: int,
    allow_busy: bool = False,
    wait_idle_s: float = 1800.0,
) -> dict[str, Any]:
    """Every case ``repeats`` times; before each case wait up to ``wait_idle_s`` for an idle machine."""
    load_before = os.getloadavg()[0]
    rows = []
    for name in cases:
        problem = wait_until_idle(not rows, 0 if allow_busy else wait_idle_s)
        if problem and not allow_busy:
            raise SystemExit(  # noqa: TRY003
                f"machine not idle before {name}: {problem}"
            )
        rows.append(
            summarise(
                name, [run_sample(name, steps, structures) for _ in range(repeats)]
            )
        )
    return {
        "repeats": repeats,
        "python": sys.version.split()[0],
        "load_average_1min_before": load_before,
        "load_average_1min_after": os.getloadavg()[0],
        "idle_at_end": idle_problem(check_load=False) is None,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """``run`` the benchmark or run one ``case`` in this process; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    case = sub.add_parser(
        "case", help="run one case in this process and print a JSON line"
    )
    case.add_argument("name")
    bench = sub.add_parser(
        "run", help="repeat the cases in fresh processes and print the table"
    )
    bench.add_argument("--cases", nargs="+", default=default_cases())
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--out", type=Path, help="also write the summary as JSON")
    bench.add_argument(
        "--wait-idle",
        type=float,
        default=1800.0,
        help="seconds to wait for an idle machine before a case (then give up)",
    )
    bench.add_argument(
        "--allow-busy",
        action="store_true",
        help="measure on a busy machine (numbers are not quotable)",
    )
    for sub_parser in (case, bench):
        sub_parser.add_argument(
            "--steps", type=int, default=100, help="steps of train_run"
        )
        sub_parser.add_argument(
            "--structures", type=int, default=100, help="structures of calc_eval"
        )
    args = parser.parse_args(argv)
    if args.command == "case":
        sys.stdout.write(
            json.dumps(run_case(args.name, args.steps, args.structures)) + "\n"
        )
        return 0
    result = run(
        args.cases,
        args.repeats,
        args.steps,
        args.structures,
        args.allow_busy,
        args.wait_idle,
    )
    sys.stdout.write(render_table(result["rows"]) + "\n")
    sys.stdout.write(
        f"load average before {result['load_average_1min_before']:.2f}, after {result['load_average_1min_after']:.2f}\n"
    )
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
