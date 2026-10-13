#!/usr/bin/env python

import os
import argparse
from collections.abc import Sequence
from tensorpotential.data.datasets import load_extxyz
from tensorpotential.formatting import sizeof_fmt

import logging
from tensorpotential.scripts._script_logging import configure_logging

logger = logging.getLogger()



def build_parser(prog: str | None = None):
    parser = argparse.ArgumentParser(prog=prog)

    parser.add_argument(
        "extxyz_filename",
        help="Name of extxyz file",
        type=str,
    )

    parser.add_argument(
        "--output-dataset-filename",
        help="pickle filename, default is inferenced from extxyz_filename",
        type=str,
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    configure_logging()
    ##############################################################################################
    parser = build_parser(prog)
    args_parse = parser.parse_args(argv)
    extxyz_filename = os.path.abspath(args_parse.extxyz_filename)
    output_dataset_filename = args_parse.output_dataset_filename
    if output_dataset_filename is None:
        output_dataset_filename = extxyz_filename.replace(".extxyz", ".pkl.gz")
        output_dataset_filename = extxyz_filename.replace(".xyz", ".pkl.gz")

    logging.info(f"Input filename: {extxyz_filename}")
    logging.info(f"Output filename: {output_dataset_filename}")

    df = load_extxyz(extxyz_filename)
    logging.info("Saving dataframe")
    df.to_pickle(output_dataset_filename, compression="gzip")
    logging.info(
        f"Saved dataframe to {output_dataset_filename} ({sizeof_fmt(output_dataset_filename)})"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
