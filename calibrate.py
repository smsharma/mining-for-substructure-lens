#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import logging
import argparse
import numpy as np

sys.path.append("./")

from inference.calibration import HistogramCalibrator
from inference.utils import s_from_r


def calibrate(
    data_dir,
    raw_filename,
    calibration_filename,
    nbins=50,
    transform_to_s=False,
    equal_binning=False,
):
    logging.info("Calibrating llr_%s.npy based on calibration data llr_%s_*.npy", raw_filename, calibration_filename)

    # Load data
    llr_raw = np.load("{}/llr_{}.npy".format(data_dir, raw_filename))
    n_grid = llr_raw.shape[0]
    logging.info("  Found %s grid points", n_grid)

    llr_calibration_den = np.load(
        "{}/llr_{}_ref.npy".format(data_dir, calibration_filename)
    )

    # Calibrate every data set
    llr_cal = np.zeros_like(llr_raw)
    for i in range(n_grid):
        try:
            llr_calibration_num = np.load(
                "{}/llr_{}_theta_{}.npy".format(data_dir, calibration_filename, i)
            )
        except FileNotFoundError:
            logging.warning("Did not find numerator calibration data for i = %s", i)
            llr_cal[i] = np.copy(llr_raw[i])

        if not np.all(np.isfinite(llr_calibration_num)):
            logging.warning("Infinite data in numerator calibration data for i = %s", i)
            llr_cal[i] = np.copy(llr_raw[i])

        if transform_to_s:
            s_cal_num = s_from_r(np.exp(llr_calibration_num))
            s_cal_den = s_from_r(np.exp(llr_calibration_den[i]))
            s_raw = s_from_r(np.exp(llr_raw[i]))

            cal = HistogramCalibrator(
                s_cal_num,
                s_cal_den,
                nbins=nbins,
                histrange=(0.0, 1.0),
                mode="fixed" if equal_binning else "dynamic",
            )

            llr_cal[i] = cal.log_likelihood_ratio(s_raw)

        else:
            cal = HistogramCalibrator(
                llr_calibration_num, llr_calibration_den[i], nbins=nbins
            )
            llr_cal[i] = cal.log_likelihood_ratio(llr_raw[i])

    llr_cal = np.array(llr_cal)

    # Save results
    np.save("{}/llr_calibrated_{}.npy".format(data_dir, raw_filename), llr_cal)

    logging.info("  Saved results at llr_calibrated_%s.npy", raw_filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strong lensing experiments: evaluation"
    )

    # Main options
    parser.add_argument("raw", type=str, help='Sample name, like "test".')
    parser.add_argument("calibration", type=str, help="File name for results.")
    parser.add_argument(
        "--bins", default=20, type=int, help="Number of bins in calibration histogram."
    )
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
    calibrate(args.dir + "/data/results/", args.raw, args.calibration, nbins=args.bins)
    logging.info("All done! Have a nice day!")
