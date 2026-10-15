"""Where the whole compiled model spends its time, per instruction, forward and backward (QUAL10).

``tools/profile_instructions.py`` times every instruction alone. Inside the whole model (energy,
forces and virial, as the calculator runs it) XLA fuses across instructions and chooses the tensor
layouts for the whole graph, so an instruction can cost several times its isolated time there.
This tool measures the whole model itself: it records a profiler trace of the compiled
``model.compute`` with one event per XLA operation (``--xla_cpu_enable_xprof_traceme``), maps each
operation to the TensorFlow op it came from through the metadata of the optimized HLO
(``--xla_dump_to``), and sums the time per instruction: ``forward`` for the ops of the
instruction, ``backward`` for the ops of its gradient (``gradient_tape/<instruction>/...``).

The operations of XLA run concurrently on several threads, so the summed op time exceeds the wall
time of a call (both are printed). Ops whose TensorFlow name does not start with an instruction
name (the force and virial sums of the model) are under ``(model)``.

Usage (from the repository root)::

    python tools/trace_model.py model_grace_2L_omat.yaml --supercell 1 --json out.json
    python tools/trace_model.py GRACE-3L-OMAT-large.yaml --top-ops 20

CPU only: the per-operation events are an option of the XLA CPU backend. The trace is read with the
protobuf module that TensorFlow ships (``tensorflow.tsl.profiler.protobuf.xplane_pb2``, TensorFlow
2.20); another TensorFlow version may move it.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_instructions as pi

MODEL = "(model)"
GRAD_PREFIX = "gradient_tape/"
# one HLO instruction per line: "%name = shape op(...), ..., metadata={... op_name="..." ...}"
HLO_LINE = re.compile(r'^\s*(?:ROOT )?%?([\w.\-]+) = (\S+) .*?op_name="([^"]*)"')
XLA_TRACE_FLAGS = "--xla_cpu_enable_xprof_traceme=true --xla_dump_hlo_as_text"


def hlo_metadata(text: str) -> dict[str, tuple[str, str]]:
    """``{hlo instruction name: (shape, TensorFlow op name)}`` of an HLO text dump."""
    meta = {}
    for line in text.splitlines():
        match = HLO_LINE.match(line)
        if match:
            meta[match.group(1)] = (match.group(2), match.group(3))
    return meta


def owner(op_name: str, instruction_names: list[str]) -> tuple[str, str]:
    """``(instruction, "forward" | "backward")`` of a TensorFlow op name.

    The instruction is the longest instruction name that the first scope of the op name equals or
    starts with followed by ``_`` (layers are scoped ``<instruction>_Linear_0``); ``MODEL`` if none.
    """
    phase = "forward"
    if op_name.startswith(GRAD_PREFIX):
        phase = "backward"
        op_name = op_name[len(GRAD_PREFIX) :]
    scope = op_name.split("/")[0]
    matches = [n for n in instruction_names if scope == n or scope.startswith(n + "_")]
    return (max(matches, key=len) if matches else MODEL), phase


def attribute(
    op_ms: dict[str, float],
    meta: dict[str, tuple[str, str]],
    instruction_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows per instruction (``forward_ms``, ``backward_ms``) and per HLO op, largest first.

    ``op_ms`` maps trace event names to ms per call; names without HLO metadata (the executor's
    own events, the ``call`` wrappers of the thunk runtime) are left out.
    """
    per = collections.defaultdict(lambda: {"forward_ms": 0.0, "backward_ms": 0.0})
    ops: list[dict[str, Any]] = []
    for event, ms in op_ms.items():
        name = event if event in meta else event.removesuffix(".clone")
        if name not in meta or name.startswith("call"):
            continue
        shape, op_name = meta[name]
        instruction, phase = owner(op_name, instruction_names)
        per[instruction][f"{phase}_ms"] += ms
        ops.append({"hlo": name, "shape": shape, "op_name": op_name, "ms": ms})
    rows = [{"instruction": k, **v} for k, v in per.items()]
    rows.sort(key=lambda r: -(r["forward_ms"] + r["backward_ms"]))
    ops.sort(key=lambda r: -float(r["ms"]))
    return rows, ops


def render(
    rows: list[dict[str, Any]], ops: list[dict[str, Any]], wall_ms: float, top_ops: int
) -> str:
    """Markdown table per instruction, the totals, and the ``top_ops`` largest HLO ops."""
    total = sum(r["forward_ms"] + r["backward_ms"] for r in rows)
    lines = [
        "| instruction | forward ms | backward ms | share % |",
        "|---|---|---|---|",
    ]
    for r in rows:
        both = r["forward_ms"] + r["backward_ms"]
        share = 100 * both / total if total else 0.0
        lines.append(
            f"| {r['instruction']} | {r['forward_ms']:.1f} | {r['backward_ms']:.1f} | {share:.1f} |"
        )
    lines.append(f"| summed op time | | | {total:.1f} ms |")
    lines.append(f"| wall time of one call | | | {wall_ms:.1f} ms |")
    if top_ops:
        lines += ["", "| HLO op | shape | TensorFlow op | ms |", "|---|---|---|---|"]
        lines += [
            f"| {o['hlo']} | {o['shape']} | {o['op_name']} | {o['ms']:.1f} |"
            for o in ops[:top_ops]
        ]
    return "\n".join(lines)


def trace_events_ms(space, calls: int) -> dict[str, float]:
    """Return the summed duration in ms per call of every event name of an ``XSpace``."""
    totals: dict[str, float] = collections.defaultdict(float)
    for plane in space.planes:
        names = plane.event_metadata
        for line in plane.lines:
            for event in line.events:
                totals[names[event.metadata_id].name] += event.duration_ps / 1e9 / calls
    return dict(totals)


def read_xspace(directory: Path):
    """Return the ``XSpace`` of the newest ``*.xplane.pb`` written under ``directory``."""
    from tensorflow.tsl.profiler.protobuf import xplane_pb2

    path = max(directory.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)
    space = xplane_pb2.XSpace()  # ty: ignore[unresolved-attribute] - a generated protobuf class
    space.ParseFromString(path.read_bytes())
    return space


def largest_optimized_hlo(directory: Path) -> str:
    """Return the text of the largest ``*after_optimizations.txt`` dump: the whole-model module."""
    dumps = list(directory.glob("*after_optimizations.txt"))
    return max(dumps, key=lambda p: p.stat().st_size).read_text()


def instruction_names(model) -> list[str]:
    """Names of the model's instructions (a list or a dict of them), in order."""
    import oracle_snapshot as osn

    return [ins.name for ins in osn._as_list(model.instructions)]


def trace_model(args: argparse.Namespace, tf, work: Path) -> dict[str, Any]:
    """Build, compile and trace the model of ``args.yaml``; return the JSON-ready result."""
    model = pi.build_model(tf, args.yaml, args.dtype, args.option)
    atoms = pi.structure(args.supercell)
    data = pi.input_batch(tf, model, atoms)
    wall_ms = pi.whole_model_ms(tf, model, data, args.repeats)  # compiles, then times
    tf.profiler.experimental.start(str(work / "trace"))
    for _ in range(args.calls):
        for out in model.compute(data).values():
            out.numpy()
    tf.profiler.experimental.stop()
    op_ms = trace_events_ms(read_xspace(work / "trace"), args.calls)
    meta = hlo_metadata(largest_optimized_hlo(work / "hlo"))
    rows, ops = attribute(op_ms, meta, instruction_names(model))
    return {
        "yaml": args.yaml,
        "dtype": args.dtype,
        "option": args.option,
        "n_atoms": len(atoms),
        "n_bonds": int(data["ind_i"].shape[0]),
        "wall_ms": wall_ms,
        "rows": rows,
        "ops": ops[: args.top_ops],
    }


def main(argv: list[str] | None = None) -> int:
    """Trace one yaml, print the tables; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("yaml", help="model yaml (a path or a name in tests/)")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--option", choices=pi.OPTIONS, default="none")
    parser.add_argument("--supercell", type=int, default=1)
    parser.add_argument(
        "--repeats", type=int, default=3, help="timed calls for the wall time"
    )
    parser.add_argument("--calls", type=int, default=3, help="traced calls (averaged)")
    parser.add_argument(
        "--top-ops", type=int, default=0, help="also list the N largest HLO ops"
    )
    parser.add_argument("--json", type=Path, help="write the result here")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="trace_model_") as tmp:
        work = Path(tmp)
        flags = f"{XLA_TRACE_FLAGS} --xla_dump_to={work / 'hlo'}"
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {flags}".strip()
        tf = pi.configure_tensorflow("cpu")
        result = trace_model(args, tf, work)
    header = f"{args.yaml}: {result['n_atoms']} atoms, {result['n_bonds']} bonds, {args.dtype}, option {args.option}, cpu"
    body = render(result["rows"], result["ops"], result["wall_ms"], args.top_ops)
    sys.stdout.write(f"\n{header}\n{body}\n")
    if args.json:
        payload = {"environment": pi.environment(tf, "cpu"), "trace": result}
        args.json.write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
