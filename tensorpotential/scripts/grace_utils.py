#!/usr/bin/env python
import argparse
import logging
from collections.abc import Sequence

from tensorpotential import _tf_options  # noqa: F401
from tensorpotential import model_ops
from tensorpotential.model_ops import load_checkpoint  # noqa: F401  (the old import path)
from tensorpotential.scripts._script_logging import configure_logging

logger = logging.getLogger()


def resolve_param_dtype(args, model_path: str):
    """Determine param_dtype: use explicit --param_dtype if given, else read from
    model.yaml metadata, else fall back to float64 for backward compatibility."""
    return model_ops.resolve_param_dtype(getattr(args, "param_dtype", None), model_path)


def preprocess_args(args):
    if not args.output_suffix.startswith("-"):
        args.output_suffix = "-" + args.output_suffix

    if args.checkpoint_path and args.checkpoint_path.endswith(".index"):
        args.checkpoint_path = args.checkpoint_path.replace(".index", "")


def _converting_kwargs(args):
    """The keyword arguments the operations that write a converted model share."""
    return {
        "potential": args.potential,
        "checkpoint_path": args.checkpoint_path,
        "output_suffix": args.output_suffix,
        "param_dtype_name": getattr(args, "param_dtype", None),
    }


def update_model(args):
    preprocess_args(args)
    model_ops.update_model(**_converting_kwargs(args))


def resave_checkpoint(args):
    preprocess_args(args)
    model_ops.resave_checkpoint(**_converting_kwargs(args))


def reduce_elements(args):
    preprocess_args(args)
    model_ops.reduce_elements(**_converting_kwargs(args), elements=args.elements)


def cast_model_param(args):
    preprocess_args(args)
    model_ops.cast_model_param(
        potential=args.potential,
        checkpoint_path=args.checkpoint_path,
        output_suffix=args.output_suffix,
        curr=args.curr,
        to=args.to,
    )


def export_model(args):
    preprocess_args(args)
    model_ops.export_model(
        potential=args.potential,
        checkpoint_path=args.checkpoint_path,
        saved_model_name=args.saved_model_name,
        sf=args.sf,
        communicated_keys=getattr(args, "communicated_keys", None),
        param_dtype_name=getattr(args, "param_dtype", None),
    )


def export_kokkos(args):
    preprocess_args(args)
    model_ops.export_kokkos(
        potential=args.potential,
        checkpoint_path=args.checkpoint_path,
        output=args.output,
        arch=args.arch,
        uq_artifacts_path=args.uq_artifacts_path,
        param_dtype_name=getattr(args, "param_dtype", None),
    )


def model_summary(args):
    preprocess_args(args)
    model_ops.model_summary(
        potential=args.potential,
        verbose=args.verbose,
        param_dtype_name=getattr(args, "param_dtype", None),
    )


def aux_model(args):
    model_ops.aux_model(
        potential=args.potential,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        communicated_keys=args.communicated_keys,
        param_dtype_name=getattr(args, "param_dtype", None),
    )


def build_parser(prog: str = 'grace_utils'):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="CLI tool for model conversions and summarization",
    )
    parser.add_argument("-p", "--potential", required=True, help="Path to model.yaml")
    parser.add_argument(
        "--param_dtype",
        required=False,
        type=str,
        default=None,
        help="Model parameters' dtype (float32 or float64). If not set, inferred from model.yaml metadata; falls back to float64 for old models.",
    )
    parser.add_argument(
        "-c", "--checkpoint-path", required=False, help="Path to checkpoint"
    )
    parser.add_argument(
        "-os",
        "--output-suffix",
        type=str,
        default="converted",
        help="Output suffix for converted",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # update_models
    parser_update = subparsers.add_parser(
        "update_model", help="Update model (model.yaml) and corresponding checkpoint."
    )
    parser_update.set_defaults(func=lambda args: update_model(args))

    # resave checkpoint
    parser_resave_checkpoint = subparsers.add_parser(
        "resave_checkpoint",
        help="Resave model's (model.yaml) checkpoint (no optimizer)",
    )
    parser_resave_checkpoint.set_defaults(func=lambda args: resave_checkpoint(args))

    # select_elements
    parser_reduce = subparsers.add_parser(
        "reduce_elements", help="Reduce elements from the model."
    )
    parser_reduce.add_argument(
        "-e", "--elements", nargs="+", required=True, help="Elements to select"
    )
    parser_reduce.set_defaults(func=lambda args: reduce_elements(args))

    # precision
    parser_precision = subparsers.add_parser(
        "cast_model_param", help="Change model's floating point precision."
    )
    parser_precision.add_argument(
        "-curr",
        required=True,
        choices=["fp32", "fp64"],
        help="Current precision type to cast from",
    )
    parser_precision.add_argument(
        "-to",
        required=True,
        choices=["fp32", "fp64"],
        help="New precision type to cast into",
    )
    parser_precision.set_defaults(func=lambda args: cast_model_param(args))

    # export model
    parser_export = subparsers.add_parser(
        "export", help="Export model to saved_model or FS/C++ format."
    )
    parser_export.add_argument(
        "-sf",
        action="store_true",
        default=False,
        help="Save to GRACE-FS/C++ YAML model format",
    )
    parser_export.add_argument(
        "-n",
        "--saved-model-name",
        dest="saved_model_name",
        type=str,
        default="saved_model",
        help="Path where to save the SavedModel or YAML file",
    )
    parser_export.add_argument(
        "-ck",
        "--communicated-keys",
        nargs="+",
        default=None,
        help="List of communicated keys for 2L model parallel split export",
    )
    parser_export.set_defaults(func=lambda args: export_model(args))

    # export to Kokkos .npz
    parser_kokkos = subparsers.add_parser(
        "export_kokkos",
        help="Export GRACE-1L/2L/3L weights to .npz for LAMMPS Kokkos pair style.",
    )
    parser_kokkos.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output .npz path (e.g. grace_2l_weights.npz)",
    )
    parser_kokkos.add_argument(
        "--arch",
        choices=["auto", "1l", "2l", "3l"],
        default="auto",
        help="Override architecture (default: auto-detect from instructions).",
    )
    parser_kokkos.add_argument(
        "--uq-artifacts",
        dest="uq_artifacts_path",
        default=None,
        help="Path to gmm_artifacts.npz (UQ). If given, dense uq_* arrays "
        "(centroids, inverse covariances, gamma thresholds, error model) are "
        "baked into the exported kokkos .npz.",
    )
    parser_kokkos.set_defaults(func=lambda args: export_kokkos(args))

    # info
    parser_summary = subparsers.add_parser("summary", help="Show info about the model")
    parser_summary.add_argument(
        "-v",
        "--verbose",
        default=0,
        type=int,
        choices=[0, 1, 2],
        help="Verbosity level: 0, 1 or 2",
    )

    parser_summary.set_defaults(func=lambda args: model_summary(args))

    # upgrade model
    parser_aux_model = subparsers.add_parser(
        "aux_model",
        help="Add compute_energy; split a 2L model (see -ck)",
        description=(
            "Save the model as a SavedModel with the extra compute function compute_energy. "
            "A GRACE-2L model is also split at the communicated keys (-ck) into "
            "forward_layer_1, backward_layer_2 and backward_layer_1."
        ),
    )
    parser_aux_model.add_argument(
        "-o", "--output-path", required=True, help="Path to save the upgraded model"
    )
    parser_aux_model.add_argument(
        "-ck",
        "--communicated-keys",
        nargs="+",
        default=["I_nl_LN", "I"],
        help="Keys communicated between the two layers of a GRACE-2L model, at which it is "
        "split (unused for other models); they must exist in the model (default: I_nl_LN I)",
    )
    parser_aux_model.set_defaults(func=lambda args: aux_model(args))

    return parser


def main(argv: Sequence[str] | None = None, *, prog: str = 'grace_utils') -> int:
    configure_logging()
    # Parse arguments and execute the corresponding function
    args = build_parser(prog).parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
