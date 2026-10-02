#!/usr/bin/env python

import os
import argparse
from tensorpotential.cli.data import load_extxyz
from tensorpotential.formatting import sizeof_fmt

import logging

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
logger = logging.getLogger()



def main(args=None):
    ##############################################################################################
    parser = argparse.ArgumentParser()

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

    args_parse = parser.parse_args(args)
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


if __name__ == "__main__":
    main()
