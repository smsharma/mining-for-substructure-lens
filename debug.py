#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import logging

sys.path.append("../")

from train import train

logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG)

logging.info("Hi!")

train(
    method="carl",
    alpha=1.,
    data_dir="/Users/johannbrehmer/work/projects/other/strong_lensing/StrongLensing-Inference/data/",
    sample_name="train",
    model_filename="debug",
    log_input=False,
    batch_size=128,
    n_epochs=1,
    optimizer="adam",
    initial_lr=0.001,
    final_lr=0.0001,
    limit_samplesize=256,
    architecture="resnet",
)

logging.info("All done! Have a nice day!")
