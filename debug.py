#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
import logging

sys.path.append("./")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.DEBUG,
)

from train import train

logging.info("Hi!")

train(
    method="alices",
    alpha=0.1,
    data_dir="./data/",
    sample_name="train",
    model_filename="debug",
    aux="z",
    architecture="resnet",
    log_input=False,
    batch_size=128,
    n_epochs=1,
    optimizer="adam",
    initial_lr=0.001,
    final_lr=0.0001,
    limit_samplesize=None,
)

logging.info("All done! Have a nice day!")
