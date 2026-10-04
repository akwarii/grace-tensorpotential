"""Probe of the TensorFlow GRACE models: variables, dtypes, where constants are created and used, timings.

For a model built from a test yaml (random weights, CPU), or from a yaml plus a real checkpoint
(``--checkpoint``, the HPC job SPEC5b), the probe records into ``probe_<label>.json``:

* ``variables``: every ``model.variables`` entry with its name, shape, dtype, trainability and the
  checkpoint key it is saved under (``tf.train.list_variables``), and the two sets compared;
* ``attributes``: every tensor-valued attribute (variables, eager tensors, numpy arrays) of every
  instruction and of the modules nested in it, with the dtype it is created in;
* ``instructions``: for each instruction, a trace of ``frwrd`` into a graph on the input it sees in
  a real forward pass: the float dtype of its operations, its casts, and for every captured
  variable and constant the operations that read it with the dtype they compute in (creation and
  use side by side);
* ``timings``: neighbour list (``extract_from_ase_atoms``) against the compiled model, per structure.

Usage (from the repository root)::

    python tools/grace_probe.py write baselines/probes
    python tools/grace_probe.py write out --yamls /path/model.yaml --label 2L_OMAT \\
        --checkpoint /path/ckpt-1 --dtypes float32
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_snapshot as osn  # noqa: E402

logger = logging.getLogger("grace_probe")

TESTS = osn.TESTS
SEED = osn.SEED
DEFAULT_DTYPES = ("float64", "float32")
DEFAULT_SUPERCELLS = (1, 2, 3)
# operations that only hold or move a value: their dtype is not a computation
PLUMBING_OPS = frozenset({
    "Const",
    "Placeholder",
    "PlaceholderWithDefault",
    "ReadVariableOp",
    "Identity",
})
SKIPPED_ATTRIBUTES = ("_init_args", "_self_")


def configure_tensorflow(threads: int = 0):
    """Import TensorFlow on CPU; ``threads`` > 0 pins the intra- and inter-op thread pools."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorpotential  # noqa: F401, I001 - selects the legacy Keras backend
    import tensorflow as tf

    if threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)
    return tf


def resolve_yaml(name: str) -> Path:
    """A yaml given by path, or by name in ``tests/``."""
    path = Path(name)
    return path if path.exists() else TESTS / name


def build_probe_model(
    yaml_path: Path, dtype: str, checkpoint: str | None = None, seed: int = SEED
):
    """The model of ``yaml_path`` with parameters of ``dtype``.

    Without a checkpoint the trainable floating variables get seeded non-zero values (as in the
    oracle snapshot); with one, the weights are read from it and every variable must be matched.
    """
    tf = configure_tensorflow()
    from tensorpotential.instructions import load_instructions
    from tensorpotential.tpmodel import TPModel

    model = TPModel(load_instructions(str(yaml_path)))
    model.build(getattr(tf, dtype))
    if checkpoint is not None:
        status = tf.train.Checkpoint(model=model).read(checkpoint)
        status.assert_existing_objects_matched()
        return model
    for var in model.trainable_variables:
        if var.dtype.is_floating:
            values = osn.seeded_values(var.name, tuple(var.shape), seed)
            var.assign(values.astype(var.dtype.as_numpy_dtype))
    return model


def checkpoint_keys(model) -> dict[int, str]:
    """``id(variable)`` -> the key it has in a checkpoint of ``tf.train.Checkpoint(model=model)``.

    The tracked-object graph is walked breadth first, as the saver does, so a variable reachable
    by several paths gets the shortest one.
    """
    import tensorflow as tf

    keys: dict[int, str] = {}
    seen = {id(model)}
    queue: deque = deque([(model, ("model",))])
    while queue:
        node, path = queue.popleft()
        for name, child in node._trackable_children().items():  # noqa: SLF001 - the saver's own walk
            if isinstance(child, tf.Variable):
                keys.setdefault(id(child), "/".join((*path, name)) + SAVED_SUFFIX)
            elif id(child) not in seen:
                seen.add(id(child))
                queue.append((child, (*path, name)))
    return keys


SAVED_SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"


def variable_records(model) -> dict[str, Any]:
    """Every variable of ``model`` with its checkpoint key, and the comparison with a saved checkpoint."""
    import tensorflow as tf

    keys = checkpoint_keys(model)
    rows: list[dict[str, Any]] = [
        {
            "name": v.name,
            "shape": list(v.shape),
            "dtype": v.dtype.name,
            "trainable": bool(v.trainable),
            "checkpoint_key": keys.get(id(v)),
        }
        for v in model.variables
    ]
    with tempfile.TemporaryDirectory() as tmp:
        prefix = tf.train.Checkpoint(model=model).write(str(Path(tmp) / "ckpt"))
        saved = {
            key: shape
            for key, shape in tf.train.list_variables(prefix)
            if key.endswith(SAVED_SUFFIX)
        }
    probed = {r["checkpoint_key"] for r in rows if r["checkpoint_key"]}
    return {
        "variables": rows,
        "n_trainable": sum(r["trainable"] for r in rows),
        "n_checkpoint_keys": len(saved),
        "keys_without_variable": sorted(set(saved) - probed),
        "variables_without_key": sorted(
            r["name"] for r in rows if r["checkpoint_key"] not in saved
        ),
    }


def _tensor_kind(value) -> str | None:
    """``variable``, ``tensor`` or ``ndarray`` for a tensor-valued attribute, else ``None``."""
    import tensorflow as tf

    if isinstance(value, tf.Variable):
        return "variable"
    if isinstance(value, tf.Tensor):
        return "tensor"
    if isinstance(value, np.ndarray):
        return "ndarray"
    return None


def _children(obj) -> Iterable[tuple[str, Any]]:
    """The attributes of a module worth walking, containers flattened to ``name[key]``."""
    for name, value in vars(obj).items():
        if name.startswith(SKIPPED_ATTRIBUTES):
            continue
        if isinstance(value, dict):
            yield from ((f"{name}[{k}]", v) for k, v in value.items())
        elif isinstance(value, list | tuple):
            yield from ((f"{name}[{i}]", v) for i, v in enumerate(value))
        else:
            yield name, value


def attribute_objects(
    instruction, keys: dict[int, str]
) -> list[tuple[dict[str, Any], Any]]:
    """``(record, object)`` for the tensor-valued attributes of ``instruction`` and of the modules nested in it."""
    import tensorflow as tf

    rows: list[tuple[dict[str, Any], Any]] = []
    seen: set[int] = set()

    def walk(obj, path: str) -> None:
        for name, value in _children(obj):
            kind = _tensor_kind(value)
            if kind is not None:
                row = {
                    "path": f"{path}/{name}",
                    "kind": kind,
                    "dtype": value.dtype.name
                    if hasattr(value.dtype, "name")
                    else str(value.dtype),
                    "shape": list(value.shape),
                    "trainable": bool(value.trainable) if kind == "variable" else None,
                    "checkpoint_key": keys.get(id(value)),
                }
                rows.append((row, value))
            elif isinstance(value, tf.Module) and id(value) not in seen:
                seen.add(id(value))
                walk(value, f"{path}/{name}")

    seen.add(id(instruction))
    walk(instruction, instruction.name)
    return rows


def _consumer_label(op) -> str:
    """``Type:dtype`` of the operation that reads a value; a cast reads ``Cast:src->dst``."""
    if op.type == "Cast":
        return f"Cast:{op.get_attr('SrcT').name}->{op.get_attr('DstT').name}"
    floats = [o.dtype.name for o in op.outputs if o.dtype.is_floating]
    return f"{op.type}:{floats[0] if floats else (op.outputs[0].dtype.name if op.outputs else 'none')}"


def _consumers(tensor) -> list[str]:
    return sorted({_consumer_label(c) for c in tensor.consumers()})


def _is_float_cast(cast: str) -> bool:
    """Whether ``cast`` (``src->dst``) goes between two floating dtypes."""
    import tensorflow as tf

    return all(tf.as_dtype(name).is_floating for name in cast.split("->"))


def _attribute_field(matches: list[str]) -> dict[str, Any]:
    """The attribute a constant is, or (equal values, equal dtype) its three shortest candidates."""
    if len(matches) == 1:
        return {"attribute": matches[0]}
    shortest = sorted(matches, key=lambda p: (len(p), p))[:3]
    return {"attribute": None, "candidates": shortest, "n_candidates": len(matches)}


def trace_summary(
    graph,
    handles: dict[str, str],
    values: dict[str, np.ndarray],
    input_keys: dict[str, str],
) -> dict[str, Any]:
    """What the operations of a traced ``frwrd`` compute in, and how captured values are read.

    ``handles`` maps the name of a captured resource placeholder to the attribute path of the
    variable, ``values`` maps attribute paths of tensors and arrays to their values (a constant
    of the graph that equals one of them in dtype, shape and value is that attribute),
    ``input_keys`` maps the name of an input placeholder to its key in the input dictionary.
    """
    import tensorflow as tf

    op_dtypes: collections.Counter = collections.Counter()
    casts: collections.Counter = collections.Counter()
    captured: list[dict[str, Any]] = []
    literals: collections.Counter = collections.Counter()
    inputs: list[dict[str, Any]] = []
    for op in graph.get_operations():
        if op.type == "Cast":
            casts[f"{op.get_attr('SrcT').name}->{op.get_attr('DstT').name}"] += 1
        if op.type == "ReadVariableOp":
            path = handles.get(op.inputs[0].name)
            captured.append({
                "attribute": path,
                "created": op.get_attr("dtype").name,
                "read_by": _consumers(op.outputs[0]),
            })
        elif op.type == "Const" and op.outputs[0].dtype.is_floating:
            value = tf.make_ndarray(op.get_attr("value"))
            match = [
                p
                for p, v in values.items()
                if v.dtype == value.dtype
                and v.shape == value.shape
                and np.array_equal(v, value)
            ]
            if match:
                captured.append({
                    **_attribute_field(match),
                    "created": value.dtype.name,
                    "read_by": _consumers(op.outputs[0]),
                })
            else:
                literals[value.dtype.name] += 1
        elif op.type == "Placeholder" and op.outputs[0].dtype.is_floating:
            readers = _consumers(op.outputs[0])
            if readers:
                inputs.append({
                    "key": input_keys.get(op.name, op.name),
                    "dtype": op.outputs[0].dtype.name,
                    "read_by": readers,
                })
        elif op.type not in PLUMBING_OPS:
            for out in op.outputs:
                if out.dtype.is_floating:
                    op_dtypes[out.dtype.name] += 1
                    break
    return {
        "op_float_dtypes": dict(op_dtypes),
        "casts": dict(casts),
        "literal_float_constants": dict(literals),
        "inputs": inputs,
        "captured": captured,
        "mixed_float_dtypes": len(op_dtypes) > 1
        or any(_is_float_cast(c) for c in casts),
    }


def _instructions_of(model) -> list:
    return osn._as_list(model.instructions)  # noqa: SLF001 - list or dict of instructions


def instruction_inputs(model, atoms) -> list[tuple[Any, dict]]:
    """``(instruction, input it sees)`` for a forward pass over ``atoms``; the input is a shallow copy."""
    tf = configure_tensorflow()
    batch = osn._geometry_batch(model, atoms)  # noqa: SLF001 - the snapshot's batch builder
    data = tf.data.Dataset.from_tensors(batch).get_single_element()
    seen = []
    for ins in _instructions_of(model):
        seen.append((ins, dict(data)))
        ins(data, training=False, local=False)
    return seen


def trace_instruction(ins, snapshot: dict, keys: dict[int, str]) -> dict[str, Any]:
    """Trace ``ins`` on ``snapshot`` into a graph and summarise it (an error is recorded, not raised)."""
    import tensorflow as tf

    pairs = attribute_objects(ins, keys)
    entry: dict[str, Any] = {"name": ins.name, "class": type(ins).__name__}
    spec = {k: tf.TensorSpec(v.shape, v.dtype) for k, v in snapshot.items()}

    def call(data):
        ins(dict(data), training=False, local=False)

    try:
        concrete = tf.function(call).get_concrete_function(spec)
    except Exception as exc:  # noqa: BLE001 - a class that cannot be traced is a finding
        entry["trace_error"] = f"{type(exc).__name__}: {str(exc).strip()[-300:]}"
        return entry
    variables = {id(v): a["path"] for a, v in pairs if a["kind"] == "variable"}
    handles = {
        internal.name: variables[id(ext_var)]
        for ext_var, internal in _variable_captures(concrete, ins)
        if id(ext_var) in variables
    }
    values = {
        a["path"]: np.asarray(v) for a, v in pairs if a["kind"] in ("tensor", "ndarray")
    }
    input_keys = {
        t.name.split(":")[0]: key
        for t, key in zip(concrete.inputs, sorted(spec), strict=False)
    }
    entry.update(trace_summary(concrete.graph, handles, values, input_keys))
    entry["attributes"] = [a for a, _ in pairs]
    return entry


def _variable_captures(concrete, ins) -> list[tuple[Any, Any]]:
    """``(variable, resource placeholder)`` for every variable captured by ``concrete``."""
    import tensorflow as tf

    by_handle = {}
    for v in ins.variables:
        by_handle[id(v.handle)] = v
    pairs = []
    for external, internal in concrete.graph.captures:
        var = by_handle.get(id(external))
        if var is not None and isinstance(var, tf.Variable):
            pairs.append((var, internal))
    return pairs


def median_ms(fn: Callable[[], Any], repeats: int) -> float:
    """Median wall time of ``fn`` in milliseconds over ``repeats`` calls."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def time_structure(model, builder, atoms, repeats: int) -> dict[str, Any]:
    """Neighbour-list time against compiled-model time for one structure (all in ms)."""
    tf = configure_tensorflow()
    row: dict[str, Any] = {"n_atoms": len(atoms)}
    row["neighbour_list_ms"] = median_ms(
        lambda: builder.join_to_batch([builder.extract_from_ase_atoms(atoms.copy())]),
        repeats,
    )
    batch = builder.join_to_batch([builder.extract_from_ase_atoms(atoms.copy())])
    row["n_bonds"] = int(batch["ind_i"].shape[0])
    data = {k: tf.convert_to_tensor(batch[k]) for k in model.compute_specs}
    first = time.perf_counter()
    model.compute(data)
    row["first_call_ms"] = (time.perf_counter() - first) * 1e3
    row["model_ms"] = median_ms(lambda: model.compute(data), repeats)
    row["neighbour_list_share"] = row["neighbour_list_ms"] / (
        row["neighbour_list_ms"] + row["model_ms"]
    )
    return row


def timing_structures(
    n_structures: int, supercells: Sequence[int]
) -> list[tuple[str, Any]]:
    """The first test structures, then supercells of the first one, as ``(label, atoms)``."""
    atoms_list = osn.load_structures(TESTS / osn.DEFAULT_STRUCTURES, n_structures)
    out = [(f"s{i}", a) for i, a in enumerate(atoms_list)]
    out += [(f"s0x{n}", atoms_list[0].repeat((n, n, n))) for n in supercells if n > 1]
    return out


def time_model(model, structures, repeats: int) -> list[dict[str, Any]]:
    """Timing rows for ``structures``; the model is compiled the way the calculator runs it."""
    from tensorpotential.data.databuilder import GeometricalDataBuilder
    from tensorpotential.tpmodel import extract_cutoff_and_elements

    model.decorate_compute_function(jit_compile=True)
    cutoff, symbols, index = extract_cutoff_and_elements(model.instructions)
    builder = GeometricalDataBuilder(
        elements_map=dict(zip(symbols, index, strict=True)), cutoff=float(cutoff)
    )
    rows = []
    for label, atoms in structures:
        row = time_structure(model, builder, atoms, repeats)
        rows.append({"structure": label, **row})
    return rows


def probe_model(model, atoms, keys_only: bool = False) -> dict[str, Any]:
    """Variables, attributes and per-instruction traces of ``model`` (forward pass on ``atoms``)."""
    keys = checkpoint_keys(model)
    report = variable_records(model)
    if keys_only:
        return report
    report["instructions"] = [
        trace_instruction(ins, snapshot, keys)
        for ins, snapshot in instruction_inputs(model, atoms)
    ]
    report["trace_errors"] = [
        i["name"] for i in report["instructions"] if "trace_error" in i
    ]
    report["mixed_float_instructions"] = [
        i["name"] for i in report["instructions"] if i.get("mixed_float_dtypes")
    ]
    return report


def environment() -> dict[str, Any]:
    """Versions, machine and load, so a timing can be judged."""
    import tensorflow as tf

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rev = None
    return {
        "git_rev": rev,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "load_average_1min_at_start": os.getloadavg()[0],
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "devices": [d.device_type for d in tf.config.list_physical_devices()],
    }


def probe_label(
    label: str,
    yaml_path: Path,
    dtypes: Sequence[str],
    checkpoint: str | None,
    settings: argparse.Namespace,
) -> dict[str, Any]:
    """The full probe of one model in each of ``dtypes``."""
    atoms = osn.load_structures(TESTS / osn.DEFAULT_STRUCTURES, 1)[0]
    result: dict[str, Any] = {
        "label": label,
        "yaml": yaml_path.name,
        "checkpoint": None if checkpoint is None else Path(checkpoint).name,
        "random_weights": checkpoint is None,
        "dtypes": {},
    }
    for dtype in dtypes:
        model = build_probe_model(yaml_path, dtype, checkpoint)
        section = probe_model(model, atoms)
        if not settings.no_timing:
            structures = timing_structures(settings.n_structures, settings.supercells)
            section["timings"] = time_model(model, structures, settings.repeats)
        result["dtypes"][dtype] = section
    result["environment"] = environment()
    return result


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser(
        "write", help="probe the models and write probe_<label>.json"
    )
    write.add_argument("out", type=Path, help="output directory")
    write.add_argument(
        "--yamls",
        nargs="+",
        default=list(osn.DEFAULT_YAMLS),
        help="yamls (paths or names in tests/)",
    )
    write.add_argument(
        "--dtypes", nargs="+", default=list(DEFAULT_DTYPES), choices=DEFAULT_DTYPES
    )
    write.add_argument("--label", help="label of a single yaml (default: its stem)")
    write.add_argument(
        "--checkpoint",
        help="checkpoint prefix to read the weights from (one yaml, one dtype)",
    )
    write.add_argument("--n-structures", type=int, default=3)
    write.add_argument(
        "--supercells", nargs="*", type=int, default=list(DEFAULT_SUPERCELLS)
    )
    write.add_argument("--repeats", type=int, default=5)
    write.add_argument(
        "--threads", type=int, default=0, help="TF thread pool size (0: TF default)"
    )
    write.add_argument("--no-timing", action="store_true")
    return parser.parse_args(argv)


def _check_checkpoint_args(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        return
    if len(args.yamls) != 1 or len(args.dtypes) != 1:
        raise SystemExit("--checkpoint needs exactly one yaml and one dtype")


def main(argv: list[str] | None = None) -> int:
    """Run ``write``; return the exit status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    _check_checkpoint_args(args)
    configure_tensorflow(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.yamls:
        path = resolve_yaml(name)
        label = args.label or path.stem
        result = probe_label(label, path, args.dtypes, args.checkpoint, args)
        target = args.out / f"probe_{label}.json"
        target.write_text(json.dumps(result, indent=1) + "\n")
        sys.stdout.write(f"{target}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
