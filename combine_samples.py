#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import numpy as np
import re

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.info,
)


def shuffle_and_combine(dir, input_samples, output_sample, regex=False):
    logger.info("Starting shuffling and combining")
    logger.info("  Folder:              %s", dir)
    logger.info("  Input samples:       %s", input_samples[0])
    for sample in input_samples[1:]:
        logger.info("                       %s", sample)
    logger.info("  Output sample:       %s", output_sample)
    logger.info("  Regular expressions: %s", regex)

    # Path and filenames
    folder = "{}/data/samples/".format(dir)
    filenames = ["theta", "x", "y", "r_xz", "t_xz", "n_subs", "m_subs"]

    # Parse regular expressions
    if regex:
        input_expressions = input_samples
        input_samples = []
        for expr in input_expressions:

            logging.debug(
                "Parsing regex %s in folder %s", "x_(" + expr + ")\.npy", folder
            )

            regex = re.compile("x_(" + expr + ")\.npy")

            for root, _, files in os.walk(folder):
                for file in files:
                    if regex.match(file):
                        input_sample = file[2:-4]

                        if input_sample in input_samples:
                            logging.debug(
                                "  Input sample %s already in list", input_sample
                            )
                            continue

                        logging.debug("  Found input sample %s", input_sample)
                        input_samples.append(input_sample)

        if len(input_samples) == 0:
            logging.warning("  No matching input samples found!")
            return

    # Clean samples
    for input_sample in input_samples:
        remove_infs_and_nans(folder, filenames, input_sample)

    # Combine samples
    n_samples = None
    permutation = None

    for filename in filenames:

        # Load individual files
        try:
            individuals = [
                np.load(folder + "/" + filename + "_" + input_sample + ".npy")
                for input_sample in input_samples
            ]
        except FileNotFoundError:
            logger.info(
                "Object %s does not exist for (some of the) input samples", filename
            )
            continue

        # Combine
        try:
            combined = np.concatenate(individuals, axis=0)
        except ValueError:
            logging.warning(
                "Object %s: individual results do not have matching shapes!", filename
            )
            for input_sample, individual in zip(input_samples, individuals):
                logging.warning(
                    "  %s: %s has shape %s", input_sample, filename, individual.shape
                )
            continue
        logger.info(
            "Combined %s %s files, combined shape: %s",
            len(individuals),
            filename,
            combined.shape,
        )

        # Shuffle
        if n_samples is None or permutation is None:
            n_samples = combined.shape[0]
            permutation = np.random.permutation(n_samples)
        else:
            if n_samples != combined.shape[0]:
                logging.error("Inconsistent shapes!")
                raise RuntimeError("Inconsistent shapes!")

        combined = combined[permutation]
        logger.info("Shuffled combined %s results", filename)

        # Save
        try:
            np.save(folder + "/" + filename + "_" + output_sample + ".npy", combined)
            np.savez_compressed(
                folder + "/" + filename + "_" + output_sample + ".npz", combined
            )
        except FileExistsError:
            logging.warning(
                "File %s already exists, cannot save results!",
                folder + "/" + filename + "_" + output_sample + ".npy",
            )
            continue
        logger.info(
            "Saved file %s", folder + "/" + filename + "_" + output_sample + ".npy"
        )


def remove_infs_and_nans(folder, filenames, input_sample):
    data = []
    out_filenames = []
    for filename in filenames:
        try:
            data.append(np.load(folder + "/" + filename + "_" + input_sample + ".npy"))
            out_filenames.append(folder + "/" + filename + "_" + input_sample + "_cleaned.npy")
        except FileNotFoundError:
            pass

    cut = None
    for array in data:
        this_cut = np.all(np.isfinite(array.reshape(array.shape[0], -1)), axis=1)
        if cut is None:
            cut = this_cut
        else:
            cut = np.logical_and(cut, this_cut)

    n_pass = np.sum(cut, dtype=np.int)
    n_fail = len(cut) - n_pass
    logger.info("Cleaning up *_%s.npy: %s samples pass, %s samples removed", folder, input_sample, n_pass, n_fail)

    for array, out_filename in zip(data, out_filenames):
        cleaned_array = array[cut]
        np.save(out_filename, cleaned_array)


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Strong lensing experiments: combining samples"
    )

    parser.add_argument("output", help='Combined sample label (like "train" or "test")')
    parser.add_argument(
        "inputs",
        nargs="+",
        help='Individual input sample labels (like "train0 train1 train2"). If '
        "option --regex is set, inputs can be regular expressions.",
    )
    parser.add_argument(
        "--regex", action="store_true", help="Allows regular expressions in inputs"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory. Samples will be looked for / saved in the data/samples subfolder.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    shuffle_and_combine(args.dir, args.inputs, args.output, args.regex)

    logger.info("All done! Have a nice day!")
