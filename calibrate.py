#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import logging
import argparse
import numpy as np

sys.path.append("./")

from inference.calibration import HistogramCalibrator


def calibrate(
    data_dir,
    raw_filename,
    calibration_filename="calibrate",
):
    # Load data
    llr_raw = np.load("{}/llr_{}.npy".format(data_dir, raw_filename))
    n_grid = llr_raw.shape[0]

    # Calibrate every data set
    llr_cal = np.zeros_like(llr_raw)
    for i in range(n_grid):
        llr_cal_num = np.load("{}/llr_{}_theta{}.npy".format(data_dir, calibration_filename, i))
        llr_cal_den = np.load("{}/llr_{}_ref.npy".format(data_dir, calibration_filename))

        cal = HistogramCalibrator(llr_cal_num, llr_cal_den)

        llr_cal[i] = cal.log_likelihood_ratio(llr_raw)

    llr_cal = np.array(llr_cal)

    # Save results
    np.save("{}/llr_calibrated_{}.npy".format(data_dir, raw_filename), llr_cal)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strong lensing experiments: evaluation"
    )

    # Main options
    parser.add_argument("filename", type=str, help='Sample name, like "test".')
    parser.add_argument("--cal", default="calibrate", type=str, help="File name for results.")
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory. Training data will be loaded from the data/samples subfolder, the model saved in the "
        "data/models subfolder.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.INFO,
    )
    logging.info("Hi!")
    args = parse_args()
    calibrate(args.dir + "/data/results/", args.filename, args.cal)
    logging.info("All done! Have a nice day!")
