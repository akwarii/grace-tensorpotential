"""Per-class cost of a TensorFlow GRACE model: each instruction timed alone, on CPU or GPU (QUAL9).

Every instruction of a model built from a yaml is jitted (XLA) on the input it sees in a real forward pass:

* ``fwd_ms``: the instruction's forward pass;
* ``fwd_bwd_ms``: forward plus the backward pass of its float outputs, seeded with fixed random upstream
  gradients, with respect to its float inputs (what the force and virial derivative ask of it at
  inference) and, with ``--grads all`` (the default, what training adds), to its variables;
* ``whole_ms``: the whole model (energy, forces, virial) as the calculator runs it, fused by XLA.

The inputs and upstream gradients are arguments of the jitted functions, not constants captured in
them: XLA would otherwise fold the work that depends on them alone, and an all-ones upstream gradient
(the gradient of a plain sum) lets it simplify most of the backward pass (QUAL10). XLA fuses across
instructions and lays tensors out for the whole graph, so the isolated times are an attribution, not
a partition: their sum differs from ``whole_ms`` (both are printed). Medians of ``--repeats`` calls
after one compiling call; each call ends by copying its outputs to the host, so a GPU timing includes
the transfer.

Usage (from the repository root)::

    python tools/profile_instructions.py model_grace_2L_omat.yaml --supercell 2 --json out.json
    python tools/profile_instructions.py model_grace.yaml --device gpu --dtype float32 --option dense_nbr

``--device gpu`` stops with an error when TensorFlow sees no GPU; the result file records the device.
Exit status 3 means the bond layout option does not apply to the model (no instruction takes it).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
STRUCTURES = TESTS / "data" / "MoNbTaW_test50.pkl.gz"
SEED = 7
NOT_APPLICABLE = 3  # exit status: the model has nothing the requested option applies to
OPTIONS = ("none", "dense_nbr", "lm_first")


class UnsupportedOptionError(ValueError):
    """The model has no instruction that takes the requested bond layout option."""


def configure_tensorflow(device: str):
    """Import TensorFlow for ``device`` (``cpu`` hides every GPU, ``gpu`` needs one) and return it."""
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorpotential  # noqa: F401, I001 - applies the TensorFlow options before TensorFlow is imported
    import tensorflow as tf

    if device == "gpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError(  # noqa: TRY003
            "--device gpu: TensorFlow sees no GPU (CUDA_VISIBLE_DEVICES, driver or tensorflow[and-cuda])"
        )
    return tf


def environment(tf, device: str) -> dict[str, Any]:
    """Versions, machine and (for ``gpu``) the device name and compute capability."""
    info: dict[str, Any] = {
        "device": device,
        "tensorflow": tf.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu":
        details = tf.config.experimental.get_device_details(gpus[0])
        info["gpu_name"] = details.get("device_name")
        info["gpu_compute_capability"] = list(details.get("compute_capability", ()))
        info["n_gpus_visible"] = len(gpus)
    return info


def build_model(tf, yaml_name: str, dtype: str, option: str):
    """Return the model of ``yaml_name`` (a path or a name in ``tests/``) with seeded random weights of ``dtype``."""
    sys.path.insert(0, str(ROOT / "tools"))
    import oracle_snapshot as osn

    from tensorpotential.instructions import load_instructions
    from tensorpotential.tpmodel import TPModel

    path = Path(yaml_name)
    path = path if path.exists() else TESTS / yaml_name
    instructions = load_instructions(str(path))
    if option != "none":
        try:
            instructions = osn.with_layout_option(instructions, option)
        except ValueError as exc:
            raise UnsupportedOptionError(f"{path.name}: {exc}") from exc  # noqa: TRY003
    model = TPModel(instructions)
    model.build(getattr(tf, dtype))
    for var in model.trainable_variables:
        if var.dtype.is_floating:
            values = osn.seeded_values(var.name, tuple(var.shape), SEED)
            var.assign(values.astype(var.dtype.as_numpy_dtype))
    return model


def structure(supercell: int):
    """Return the largest structure of the test set, repeated ``supercell`` times along each axis."""
    import pandas as pd

    frame = pd.read_pickle(STRUCTURES)  # noqa: S301 - repo test data
    largest = max(list(frame["ase_atoms"]), key=len)
    return largest.repeat((supercell,) * 3)


def input_batch(tf, model, atoms) -> dict:
    """Return the model input of ``atoms`` in the bond layout the model expects, as tensors."""
    sys.path.insert(0, str(ROOT / "tools"))
    import oracle_snapshot as osn

    batch = osn._geometry_batch(model, atoms)  # the snapshot's batch builder
    return {k: tf.convert_to_tensor(batch[k]) for k in model.compute_specs}


def instruction_inputs(model, data: dict) -> list[tuple[Any, dict]]:
    """``(instruction, input it sees)`` along a forward pass over ``data`` (a shallow copy each)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import oracle_snapshot as osn

    seen = []
    running = dict(data)
    for ins in osn._as_list(model.instructions):  # list or dict of instructions
        seen.append((ins, dict(running)))
        ins(running, training=False, local=False)
    return seen


def instruction_functions(
    tf, ins, snapshot: dict, with_variables: bool = True
) -> tuple[Callable, Callable, tuple, tuple]:
    """``(forward, forward_backward, forward_args, backward_args)`` of ``ins`` on ``snapshot``.

    The functions are plain Python (the caller jits them) and take their inputs as arguments:
    ``forward(floats, rest)`` and ``forward_backward(floats, rest, seeds)``, where ``seeds`` are
    fixed random upstream gradients of the float outputs (``SEED``).
    """
    floats = {k: v for k, v in snapshot.items() if v.dtype.is_floating}
    rest = {k: v for k, v in snapshot.items() if not v.dtype.is_floating}
    input_keys = set(snapshot)

    def outputs(data: dict) -> list:
        return [data[k] for k in sorted(set(data) - input_keys)]

    def float_outputs(data: dict) -> list:
        return [o for o in tf.nest.flatten(outputs(data)) if o.dtype.is_floating]

    def forward(inputs: dict, others: dict) -> list:
        data = {**inputs, **others}
        ins(data, training=False, local=False)
        return outputs(data)

    def forward_backward(inputs: dict, others: dict, seeds: list) -> list:
        with tf.GradientTape() as tape:
            tape.watch(list(inputs.values()))
            data = {**inputs, **others}
            ins(data, training=False, local=False)
            outs = float_outputs(data)
        sources = [*inputs.values(), *(ins.variables if with_variables else [])]
        grads = tape.gradient(outs, sources, output_gradients=seeds) if outs else []
        return [tf.convert_to_tensor(g) for g in grads if g is not None] or [
            tf.constant(0.0)
        ]

    data = {**floats, **rest}
    ins(data, training=False, local=False)
    rng = np.random.default_rng(SEED)
    seeds = [
        tf.constant(rng.standard_normal(o.shape), o.dtype) for o in float_outputs(data)
    ]
    return forward, forward_backward, (floats, rest), (floats, rest, seeds)


def median_ms(tf, fn: Callable, repeats: int) -> float:
    """Median wall time in ms of ``fn`` (compiled by one first call), outputs copied to the host."""

    def run() -> None:
        for out in tf.nest.flatten(fn()):
            out.numpy()

    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def timed_or_error(
    tf, fn: Callable, repeats: int, args: tuple = ()
) -> tuple[float | None, str | None]:
    """``(median ms, None)`` of ``fn(*args)`` jitted, or ``(None, error)`` for an instruction XLA cannot compile."""
    compiled = tf.function(fn, jit_compile=True)
    try:
        return median_ms(tf, lambda: compiled(*args), repeats), None
    except (tf.errors.OpError, ValueError, TypeError) as exc:
        return None, f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:160]}"


def profile_instructions(
    tf, model, data: dict, repeats: int, with_variables: bool = True
) -> list[dict[str, Any]]:
    """One row per instruction: name, class, ``fwd_ms``, ``fwd_bwd_ms`` and any error."""
    rows = []
    for ins, snapshot in instruction_inputs(model, data):
        forward, forward_backward, f_args, b_args = instruction_functions(
            tf, ins, snapshot, with_variables
        )
        fwd, err_f = timed_or_error(tf, forward, repeats, f_args)
        both, err_b = timed_or_error(tf, forward_backward, repeats, b_args)
        rows.append({
            "name": ins.name,
            "class": type(ins).__name__,
            "fwd_ms": fwd,
            "fwd_bwd_ms": both,
            "error": err_f or err_b,
        })
    return rows


def whole_model_ms(tf, model, data: dict, repeats: int) -> float:
    """Median ms of the whole compiled model (energy, forces, virial), as the calculator runs it."""
    model.decorate_compute_function(jit_compile=True)
    return median_ms(tf, lambda: list(model.compute(data).values()), repeats)


def add_shares(rows: list[dict[str, Any]]) -> None:
    """Add ``share`` (percent of the summed ``fwd_bwd_ms``) to every row that has one."""
    total = sum(r["fwd_bwd_ms"] for r in rows if r["fwd_bwd_ms"] is not None)
    for row in rows:
        ok = row["fwd_bwd_ms"] is not None and total > 0
        row["share"] = 100 * row["fwd_bwd_ms"] / total if ok else None


def _cell(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def render(rows: list[dict[str, Any]], whole_ms: float, top: int | None = None) -> str:
    """Return the markdown table, largest ``fwd_bwd_ms`` first, and the two totals."""
    ordered = sorted(rows, key=lambda r: -(r["fwd_bwd_ms"] or 0.0))[:top]
    lines = [
        "| instruction | class | fwd ms | fwd+bwd ms | share % |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['name']} | {r['class']} | {_cell(r['fwd_ms'], '.2f')} | {_cell(r['fwd_bwd_ms'], '.2f')} | {_cell(r['share'], '.1f')} |"
        for r in ordered
    ]
    fwd = sum(r["fwd_ms"] for r in rows if r["fwd_ms"] is not None)
    both = sum(r["fwd_bwd_ms"] for r in rows if r["fwd_bwd_ms"] is not None)
    lines.append(f"| sum of isolated | | {fwd:.2f} | {both:.2f} | |")
    lines.append(
        f"| whole model, fused (energy, forces, virial) | | | {whole_ms:.2f} | |"
    )
    return "\n".join(lines)


def peak_rss_mb(status: Path = Path("/proc/self/status")) -> float | None:
    """Return the peak resident memory of this process in MB, or ``None`` without ``/proc``."""
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return None


def profile_model(args: argparse.Namespace, tf, yaml_name: str) -> dict[str, Any]:
    """Return the full profile of one yaml as a JSON-ready dict."""
    model = build_model(tf, yaml_name, args.dtype, args.option)
    atoms = structure(args.supercell)
    data = input_batch(tf, model, atoms)
    whole_only = getattr(args, "whole_only", False)
    rows = (
        []
        if whole_only
        else profile_instructions(tf, model, data, args.repeats, args.grads == "all")
    )
    add_shares(rows)
    result: dict[str, Any] = {
        "yaml": yaml_name,
        "dtype": args.dtype,
        "option": args.option,
        "n_atoms": len(atoms),
        "n_bonds": int(data["ind_i"].shape[0]),
        "repeats": args.repeats,
        "grads": args.grads,
        "rows": rows,
    }
    if whole_only and args.device == "gpu":
        # the peak of the whole model alone: weights, input and the compiled computation
        tf.config.experimental.reset_memory_stats("GPU:0")
    result["whole_ms"] = whole_model_ms(tf, model, data, args.repeats)
    if args.device == "gpu":
        info = tf.config.experimental.get_memory_info("GPU:0")
        result["gpu_peak_memory_mb"] = info["peak"] / 2**20
    result["peak_rss_mb"] = peak_rss_mb()
    return result


def main(argv: list[str] | None = None) -> int:
    """Profile each yaml and print the tables; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "yamls", nargs="+", help="model yamls (paths or names in tests/)"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument(
        "--option", choices=OPTIONS, default="none", help="bond layout option"
    )
    parser.add_argument(
        "--grads",
        choices=("all", "inputs"),
        default="all",
        help="backward w.r.t. inputs and variables (training) or inputs only (inference)",
    )
    parser.add_argument(
        "--supercell", type=int, default=2, help="repeats of the 54-atom cell per axis"
    )
    parser.add_argument(
        "--whole-only",
        action="store_true",
        help="skip the per-class rows: time and memory of the whole model only",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--top", type=int, help="print only the N largest rows")
    parser.add_argument(
        "--json", type=Path, help="write every profile and the environment here"
    )
    args = parser.parse_args(argv)
    tf = configure_tensorflow(args.device)
    results = []
    for name in args.yamls:
        try:
            result = profile_model(args, tf, name)
        except UnsupportedOptionError as exc:
            sys.stderr.write(f"not applicable: {exc}\n")
            return NOT_APPLICABLE
        results.append(result)
        header = f"{name}: {result['n_atoms']} atoms, {result['n_bonds']} bonds, {args.dtype}, option {args.option}, {args.device}"
        body = (
            f"whole model {result['whole_ms']:.1f} ms"
            if args.whole_only
            else render(result["rows"], result["whole_ms"], args.top)
        )
        sys.stdout.write(f"\n{header}\n{body}\n")
    if args.json:
        payload = {"environment": environment(tf, args.device), "profiles": results}
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
