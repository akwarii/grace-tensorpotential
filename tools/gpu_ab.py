"""Before/after comparison of two commits of the TensorFlow model on a GPU, restartable and parallel (QUAL10).

Both commits are measured by the same tools (``tools/profile_instructions.py`` of this tree); only the
``tensorpotential`` package differs, extracted from each commit with ``git archive`` into
``OUT/trees/<sha>/`` and put first on ``PYTHONPATH`` (checked: ``tensorpotential.__file__`` must lie in
it). Runs of the two commits alternate (A, B, A, B, ...) so that both see the same GPU state.

The work is split into independent **steps** (``--list`` prints them):

* ``infer-<arch>-<dtype>-<option>``: whole model (energy, forces, virial), ``pairs`` alternating runs;
* ``train-<arch>-<dtype>``: a jitted training step (energy and forces loss, gradient of every weight);
* ``scan-<arch>-<dtype>-<option>``: the whole model at growing supercells, one run per size and commit,
  until a size fails (out of memory): the largest size that fits and the peak memory per bond;
* ``numerics-<arch>-<dtype>``: the outputs of both commits on the same structure, compared in the summary.

**Checkpoints.** Every run writes its own result file under ``OUT/runs/<step>/`` (``<side>-<n>.json``,
or ``.failed`` / ``.na``); a step that is started again skips the runs it already has, and a finished
step leaves ``OUT/done/<step>``. The first call writes ``OUT/config.json`` (commits resolved to their
hashes, the settings); later calls on the same ``OUT`` use it and ignore other settings. So an
interrupted job is resumed by running the same command again, and steps can run at the same time on
different GPUs with the same ``OUT`` (``--step NAME``; the extraction of the trees is locked).

``--summary`` writes ``OUT/summary.md`` and ``OUT/summary.json`` from whatever results exist:
medians per commit, the change, a ``WORSE`` flag above the threshold (5%, QUAL10), the largest size
that fits, the numeric differences; ``OUT/environment.txt`` records the GPU, the driver and the
versions.

Usage (from the repository root)::

    python tools/gpu_ab.py OUT --base df8d556 --head HEAD --list
    python tools/gpu_ab.py OUT                      # every step not done yet, then the summary
    python tools/gpu_ab.py OUT --step infer-GRACE-3L-OMAT-large-float64-none
    python tools/gpu_ab.py OUT --summary
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import itertools
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "tools" / "profile_instructions.py"
SIDES = ("base", "head")
NOT_APPLICABLE = (
    3  # exit status of profile_instructions: the option does not apply to the model
)
WORSE_THRESHOLD = 0.05
ARCHS = (
    "GRACE-FS-OMAT",
    "GRACE-1L-OMAT-large-base",
    "GRACE-2L-OMAT-large-base",
    "GRACE-3L-OMAT-large",
)


@dataclass(frozen=True)
class Config:
    """Settings of one comparison; written once to ``OUT/config.json``."""

    base: str
    head: str
    device: str = "gpu"
    pairs: int = 5
    repeats: int = 5
    cell: int = 1
    archs: tuple[str, ...] = ARCHS
    dtypes: tuple[str, ...] = ("float64", "float32")
    options: tuple[str, ...] = ("none", "dense_nbr")
    scan_cells: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
    numerics_cell: int = 1
    train_cell: int = 1
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Config:
        """Rebuild a config from its JSON form (tuples come back as lists)."""
        tuples = {"archs", "dtypes", "options", "scan_cells"}
        values: dict[str, Any] = {
            k: tuple(v) if k in tuples else v for k, v in data.items()
        }
        return cls(**values)


@dataclass(frozen=True)
class Run:
    """One measurement: a commit (``side``), what to run, and where its result goes."""

    step: str
    side: str
    index: int
    yaml: str
    dtype: str
    option: str
    cell: int
    repeats: int
    extra: tuple[str, ...] = ()


def resolve(ref: str, root: Path = ROOT) -> str:
    """Return the full hash of ``ref`` in the repository at ``root``."""
    out = subprocess.run(  # noqa: S603 - fixed git command on a local repository
        ["git", "-C", str(root), "rev-parse", "--verify", f"{ref}^{{commit}}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def load_config(out: Path, requested: Config) -> Config:
    """Return the config of ``out``: the saved one if any (a different request is reported), else ``requested``.

    The first call writes it; parallel first calls (an array job) are serialised by a lock.
    """
    path = out / "config.json"
    with locked(out / "config.lock"):
        if not path.exists():
            path.write_text(json.dumps(asdict(requested), indent=1) + "\n")
            return requested
    saved = Config.from_json(json.loads(path.read_text()))
    if saved != requested:
        sys.stderr.write(
            f"using the settings saved in {path}; the others are ignored\n"
        )
    return saved


def step_names(config: Config) -> list[str]:
    """Every step of ``config``, in the order a single job runs them."""
    names = []
    for arch, dtype, option in itertools.product(
        config.archs, config.dtypes, config.options
    ):
        names.append(f"infer-{arch}-{dtype}-{option}")
    names += [
        f"train-{a}-{d}" for a, d in itertools.product(config.archs, config.dtypes)
    ]
    names += [
        f"scan-{a}-{d}-{o}"
        for a, d, o in itertools.product(config.archs, config.dtypes, config.options)
    ]
    names += [
        f"numerics-{a}-{d}" for a, d in itertools.product(config.archs, config.dtypes)
    ]
    return names


def parse_step(name: str, config: Config) -> tuple[str, str, str, str]:
    """``(kind, arch, dtype, option)`` of a step name of ``config``."""
    kind, rest = name.split("-", 1)
    for arch, dtype in itertools.product(config.archs, config.dtypes):
        prefix = f"{arch}-{dtype}"
        if rest == prefix:
            return kind, arch, dtype, "none"
        for option in config.options:
            if rest == f"{prefix}-{option}":
                return kind, arch, dtype, option
    raise ValueError(f"unknown step {name!r}; see --list")  # noqa: TRY003


def yaml_of(arch: str) -> str:
    """Return a model path as given, or the yaml of an architecture name in tests/data/model_yamls."""
    path = Path(arch)
    return str(
        path
        if path.suffix == ".yaml"
        else ROOT / "tests" / "data" / "model_yamls" / f"{arch}.yaml"
    )


# ------------------------------------------------------------------ trees


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created if needed) for the duration of the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def package_location(tree: Path, python: str = sys.executable) -> str:
    """Where ``import tensorpotential`` resolves with ``tree`` first on ``PYTHONPATH``."""
    env = {**os.environ, "PYTHONPATH": str(tree), "TF_CPP_MIN_LOG_LEVEL": "3"}
    out = subprocess.run(  # noqa: S603 - the interpreter running this tool
        [python, "-c", "import tensorpotential; print(tensorpotential.__file__)"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        cwd=tempfile.gettempdir(),
    )
    return out.stdout.strip().splitlines()[-1]


def ensure_tree(out: Path, sha: str, root: Path = ROOT) -> Path:
    """Extract ``tensorpotential/`` of commit ``sha`` into ``OUT/trees/<sha>`` once, and check the import."""
    out = out.resolve()  # an absolute PYTHONPATH: the runs and the check do not start in the same directory
    tree = out / "trees" / sha
    with locked(out / "trees.lock"):
        if not (tree / ".complete").exists():
            shutil.rmtree(tree, ignore_errors=True)
            tree.mkdir(parents=True)
            git_archive = [
                shutil.which("git") or "git",
                "-C",
                str(root),
                "archive",
                "--format=tar",
                sha,
                "tensorpotential",
            ]
            archive = subprocess.run(git_archive, capture_output=True, check=True)  # noqa: S603 - local git
            with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
                tar.extractall(tree, filter="data")
            location = package_location(tree)
            if not location.startswith(str(tree)):
                raise RuntimeError(  # noqa: TRY003
                    f"with PYTHONPATH={tree}, tensorpotential is imported from {location}: "
                    "an installed copy takes precedence, so both commits would run the same code"
                )
            (tree / ".complete").write_text(location + "\n")
    return tree


# ------------------------------------------------------------------ runs


def runs_of(name: str, config: Config) -> list[Run]:
    """Return the runs of one step, in the order they are taken (the two commits alternate)."""
    kind, arch, dtype, option = parse_step(name, config)
    yaml = yaml_of(arch)
    if kind == "infer":
        return [
            Run(name, side, i, yaml, dtype, option, config.cell, config.repeats)
            for i in range(config.pairs)
            for side in SIDES
        ]
    if kind == "train":
        return [
            Run(
                name,
                side,
                i,
                yaml,
                dtype,
                "none",
                config.train_cell,
                config.repeats,
                ("--train-step",),
            )
            for i in range(config.pairs)
            for side in SIDES
        ]
    if kind == "scan":
        return [
            Run(name, side, cell, yaml, dtype, option, cell, min(config.repeats, 3))
            for cell in config.scan_cells
            for side in SIDES
        ]
    if kind == "numerics":
        return [
            Run(
                name,
                side,
                0,
                yaml,
                dtype,
                "none",
                config.numerics_cell,
                1,
                ("--outputs",),
            )
            for side in SIDES
        ]
    raise ValueError(f"unknown step kind {kind!r}")  # noqa: TRY003


def result_path(out: Path, run: Run, suffix: str) -> Path:
    """Where the result of ``run`` goes, with ``suffix`` ``.json``, ``.failed``, ``.na`` or ``.npz``."""
    return out / "runs" / run.step / f"{run.side}-{run.index}{suffix}"


def recorded(out: Path, run: Run) -> str | None:
    """``"ok"``, ``"failed"`` or ``"na"`` if ``run`` already has a result, else ``None``."""
    for suffix, status in ((".json", "ok"), (".failed", "failed"), (".na", "na")):
        if result_path(out, run, suffix).exists():
            return status
    return None


def command(run: Run, config: Config, json_path: Path, python: str) -> list[str]:
    """Return the ``profile_instructions.py`` command line of ``run``."""
    cmd = [
        python,
        str(PROFILE),
        run.yaml,
        "--device",
        config.device,
        "--dtype",
        run.dtype,
        "--option",
        run.option,
        "--supercell",
        str(run.cell),
        "--repeats",
        str(run.repeats),
        "--whole-only",
        "--json",
        str(json_path),
    ]
    if "--train-step" in run.extra:
        cmd.append("--train-step")
    if "--outputs" in run.extra:
        cmd += ["--outputs", str(json_path.with_suffix(".npz"))]
    return cmd


Executor = Callable[[list[str], dict[str, str]], subprocess.CompletedProcess]


def execute(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run one measurement in its own process."""
    return subprocess.run(  # noqa: S603 - this repository's tool
        cmd, capture_output=True, text=True, env=env, check=False, cwd=ROOT
    )


def take(
    run: Run,
    config: Config,
    out: Path,
    trees: dict[str, Path],
    executor: Executor = execute,
) -> str:
    """Take ``run`` unless it has a result; record and return its status."""
    status = recorded(out, run)
    if status is not None:
        return status
    final = result_path(out, run, ".json")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_suffix(".json.partial")
    env = {
        **os.environ,
        "PYTHONPATH": str(trees[run.side]),
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }
    if config.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    done = executor(command(run, config, partial, sys.executable), env)
    if done.returncode == 0 and partial.exists():
        npz = partial.with_suffix(".npz")
        if npz.exists():
            npz.replace(result_path(out, run, ".npz"))
        partial.replace(final)  # last: the .json marks the run as done
        return "ok"
    tail = (done.stderr or "")[-3000:]
    suffix = ".na" if done.returncode == NOT_APPLICABLE else ".failed"
    result_path(out, run, suffix).write_text(f"exit {done.returncode}\n{tail}\n")
    return "na" if suffix == ".na" else "failed"


def run_step(
    name: str,
    config: Config,
    out: Path,
    trees: dict[str, Path],
    executor: Executor = execute,
) -> None:
    """Take every run of step ``name`` not taken yet, then mark the step done.

    In a ``scan`` step a commit that fails at a size (out of memory) is not run at the larger sizes;
    an option that does not apply (``na``) ends the step for both commits.
    """
    if (out / "done" / name).exists():
        return
    stopped: set[str] = set()
    for run in runs_of(name, config):
        if run.side in stopped:
            continue
        status = take(run, config, out, trees, executor)
        sys.stdout.write(f"{name} {run.side} {run.index}: {status}\n")
        if status == "na":
            stopped.update(SIDES)
        elif status == "failed" and name.startswith("scan-"):
            stopped.add(run.side)
    (out / "done").mkdir(exist_ok=True)
    (out / "done" / name).write_text("")


# ------------------------------------------------------------------ summary


def _results(out: Path, step: str, side: str) -> list[dict[str, Any]]:
    folder = out / "runs" / step
    if not folder.exists():
        return []
    results = []
    for path in sorted(folder.glob(f"{side}-*.json")):
        payload = json.loads(path.read_text())
        results.append({
            "index": int(path.stem.split("-")[-1]),
            **payload["profiles"][0],
        })
    return results


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def compare_times(out: Path, step: str, key: str) -> dict[str, Any]:
    """Median of ``key`` per commit over the runs of ``step``, the relative change and the flag."""
    row: dict[str, Any] = {"step": step}
    for side in SIDES:
        values = [r[key] for r in _results(out, step, side) if key in r]
        row[f"{side}_ms"] = _median(values)
        row[f"{side}_n"] = len(values)
        peaks = [
            r["gpu_peak_memory_mb"]
            for r in _results(out, step, side)
            if "gpu_peak_memory_mb" in r
        ]
        row[f"{side}_peak_mb"] = max(peaks) if peaks else None
    base, head = row["base_ms"], row["head_ms"]
    row["change"] = None if not base or head is None else head / base - 1
    row["worse"] = row["change"] is not None and row["change"] > WORSE_THRESHOLD
    return row


def compare_scan(out: Path, step: str) -> dict[str, Any]:
    """Largest size that fits per commit and the peak memory per bond at the largest common size."""
    row: dict[str, Any] = {"step": step}
    by_side = {
        side: {r["index"]: r for r in _results(out, step, side)} for side in SIDES
    }
    for side in SIDES:
        cells = sorted(by_side[side])
        row[f"{side}_largest_cell"] = cells[-1] if cells else None
    common = sorted(set(by_side["base"]) & set(by_side["head"]))
    if common:
        cell = common[-1]
        row["common_cell"] = cell
        for side in SIDES:
            r = by_side[side][cell]
            peak = r.get("gpu_peak_memory_mb", r.get("peak_rss_mb"))
            row[f"{side}_kb_per_bond"] = (
                None if peak is None else 1024 * peak / r["n_bonds"]
            )
            row[f"{side}_ms"] = r.get("whole_ms")
        base, head = row["base_kb_per_bond"], row["head_kb_per_bond"]
        row["change"] = None if not base or head is None else head / base - 1
        row["worse"] = row["change"] is not None and row["change"] > WORSE_THRESHOLD
    return row


def compare_numerics(out: Path, step: str) -> dict[str, Any]:
    """Largest absolute and relative (to the array maximum) difference of the outputs of the two commits."""
    import numpy as np

    paths = {side: out / "runs" / step / f"{side}-0.npz" for side in SIDES}
    row: dict[str, Any] = {"step": step}
    if not all(p.exists() for p in paths.values()):
        return row
    base, head = (np.load(paths[side]) for side in SIDES)
    worst_abs, worst_rel, worst_key = 0.0, 0.0, None
    for key in sorted(set(base.files) & set(head.files)):
        a, b = base[key].astype(np.float64), head[key].astype(np.float64)
        diff = float(np.abs(a - b).max()) if a.size else 0.0
        scale = float(np.abs(a).max()) if a.size else 0.0
        rel = diff / scale if scale > 0 else 0.0
        if rel >= worst_rel:
            worst_abs, worst_rel, worst_key = diff, rel, key
    row.update({
        "max_abs_diff": worst_abs,
        "max_scaled_diff": worst_rel,
        "key": worst_key,
    })
    return row


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _pct(change: float | None) -> str:
    return "n/a" if change is None else f"{100 * change:+.1f}%"


def render_summary(config: Config, rows: dict[str, list[dict[str, Any]]]) -> str:
    """Markdown tables of the comparison."""
    lines = [
        f"base `{config.base[:10]}` (A) against head `{config.head[:10]}` (B), device {config.device}, "
        f"{config.pairs} alternating pairs, medians; WORSE = B more than {WORSE_THRESHOLD:.0%} above A",
        "",
        "| step | A ms | B ms | change | A peak MB | B peak MB | runs A/B | |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r['step']} | {_fmt(r['base_ms'], '.3g')} | {_fmt(r['head_ms'], '.3g')} | "
        f"{_pct(r['change'])} | "
        f"{_fmt(r['base_peak_mb'], '.0f')} | {_fmt(r['head_peak_mb'], '.0f')} | "
        f"{r['base_n']}/{r['head_n']} | {'WORSE' if r['worse'] else ''} |"
        for r in rows["times"]
    )
    lines += [
        "",
        "| scan | largest cell A | largest cell B | at cell | A ms | B ms | A KB/bond | B KB/bond | change | |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows["scans"]:
        lines.append(
            f"| {r['step']} | {r.get('base_largest_cell') or 'n/a'} | {r.get('head_largest_cell') or 'n/a'} | "
            f"{r.get('common_cell', 'n/a')} | {_fmt(r.get('base_ms'), '.3g')} | "
            f"{_fmt(r.get('head_ms'), '.3g')} | {_fmt(r.get('base_kb_per_bond'), '.1f')} | "
            f"{_fmt(r.get('head_kb_per_bond'), '.1f')} | "
            f"{_pct(r.get('change'))} | "
            f"{'WORSE' if r.get('worse') else ''} |"
        )
    lines += [
        "",
        "| numerics | max abs diff | max diff / array max | key |",
        "|---|---|---|---|",
    ]
    for r in rows["numerics"]:
        lines.append(
            f"| {r['step']} | {_fmt(r.get('max_abs_diff'), '.3g')} | "
            f"{_fmt(r.get('max_scaled_diff'), '.3g')} | {r.get('key', 'n/a')} |"
        )
    return "\n".join(lines) + "\n"


def environment_text(config: Config) -> str:
    """Return the commits, interpreter, platform and, on a GPU, the output of ``nvidia-smi``."""
    lines = [
        f"base: {config.base}",
        f"head: {config.head}",
        f"python: {sys.executable} {platform.python_version()}",
        f"host: {platform.node()}",
    ]
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi], capture_output=True, text=True, check=False)  # noqa: S603
        lines += ["", out.stdout]
    return "\n".join(lines) + "\n"


def write_summary(config: Config, out: Path) -> dict[str, list[dict[str, Any]]]:
    """Write ``summary.md``, ``summary.json`` and ``environment.txt`` from the results so far."""
    names = step_names(config)
    rows = {
        "times": [
            compare_times(out, n, "whole_ms") for n in names if n.startswith("infer-")
        ]
        + [
            compare_times(out, n, "train_step_ms")
            for n in names
            if n.startswith("train-")
        ],
        "scans": [compare_scan(out, n) for n in names if n.startswith("scan-")],
        "numerics": [
            compare_numerics(out, n) for n in names if n.startswith("numerics-")
        ],
    }
    (out / "summary.md").write_text(render_summary(config, rows))
    (out / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out / "environment.txt").write_text(environment_text(config))
    return rows


# ------------------------------------------------------------------ command line


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line of the tool."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "out", type=Path, help="output directory (holds the checkpoints)"
    )
    parser.add_argument("--base", default="df8d556", help="commit A (the reference)")
    parser.add_argument("--head", default="HEAD", help="commit B")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--cell",
        type=int,
        default=1,
        help="supercell of the timed runs (54 atoms each)",
    )
    parser.add_argument("--archs", nargs="+", default=list(ARCHS))
    parser.add_argument("--dtypes", nargs="+", default=["float64", "float32"])
    parser.add_argument("--options", nargs="+", default=["none", "dense_nbr"])
    parser.add_argument(
        "--scan-cells", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8]
    )
    parser.add_argument(
        "--list", action="store_true", help="print the steps and their state"
    )
    parser.add_argument("--step", nargs="+", help="run only these steps")
    parser.add_argument("--summary", action="store_true", help="only write the summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the requested steps (all by default) and write the summary; return the exit status."""
    args = parse_args(argv)
    args.out = args.out.resolve()
    requested = Config(
        base=resolve(args.base),
        head=resolve(args.head),
        device=args.device,
        pairs=args.pairs,
        repeats=args.repeats,
        cell=args.cell,
        archs=tuple(args.archs),
        dtypes=tuple(args.dtypes),
        options=tuple(args.options),
        scan_cells=tuple(args.scan_cells),
        labels={"base": args.base, "head": args.head},
    )
    config = load_config(args.out, requested)
    names = step_names(config)
    if args.list:
        for name in names:
            state = "done" if (args.out / "done" / name).exists() else "todo"
            sys.stdout.write(f"{name}\t{state}\n")
        return 0
    if not args.summary:
        trees = {
            "base": ensure_tree(args.out, config.base),
            "head": ensure_tree(args.out, config.head),
        }
        for name in args.step or names:
            parse_step(name, config)  # an unknown name stops before any run
        for name in args.step or names:
            run_step(name, config, args.out, trees, execute)
    write_summary(config, args.out)
    sys.stdout.write((args.out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
