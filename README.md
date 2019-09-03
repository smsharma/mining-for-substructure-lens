# Inferring dark matter substructure with machine learning

Code repository for the paper
[**Mining for Dark Matter Substructure: Inferring subhalo population properties from strong lenses with machine learning**](http://https://arxiv.org/abs/1909.XXXXX)
by Johann Brehmer, Siddarth Mishra-Sharma, Joeri Hermans, Gilles Louppe, and Kyle Cranmer.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dark matter](https://img.shields.io/badge/Matter-Dark-black.svg)](./)

![Visualization of Bayesian inference on substructure properties.](figures/live_inference_with_images_reverse_small.gif)


## Abstract

The subtle and unique imprint of dark matter substructure on extended arcs in strong lensing systems contains a wealth
of information about the properties and distribution of dark matter on small scales and, consequently, about the
underlying particle physics. However, teasing out this effect poses a significant challenge since for realistic
simulations the likelihood function of population-level parameters is intractable. We apply recently-developed
simulation-based techniques to the problem of substructure inference in galaxy-galaxy strong lenses. By leveraging
additional information extracted from the simulator, neural networks are trained to estimate likelihood ratios
associated with population-level parameters characterizing substructure. We show through proof-of-principle application
to simulated data that these methods can provide an efficient and principled way to concurrently analyze an ensemble of
strong lenses, and can be used to mine the large sample of lensing images deliverable by near-future surveys for
signatures of dark matter substructure.


## Results

In [figures/](figures/) we collect the figures shown in the paper. The folder also contains a few additional plots and
animations:

- [deflection_maps.pdf](figures/deflection_maps.pdf) shows an example simulated lens including the subhalos,
the deflection map from the host halo, and the deflection map from all subhalos combined.
- [calibration_curves.pdf](figures/calibration_curves.pdf) shows calibration curves (raw network output vs calibrated
likelihood ratio) for several randomly chosen parameter points.
- [live_inference_population.gif](figures/live_inference_population.gif) shows the evolution of the posterior on the
population-level parameters with successive simulated observations.
- [live_inference_shmf.gif](figures/live_inference_shmf.gif) shows the evolution of the posterior on the
subhalo mass function with successive simulated observations.
- [live_inference_both.gif](figures/live_inference_both.gif) and
[live_inference_with_images.gif](figures/live_inference_with_images.gif) combine the evolution of both posteriors,
the latter (shown at the top of this readme) also includes the simulated observed lensed images.


## Code

The dependencies of our code are listed in [environments.yml](environment.yml).

The starting point of our code based are five high-level scripts:

- [simulate.py](simulate.py) starts the simulation of lensing systems and, depending on the command-line parameters,
creates training, calibration, or test samples.
- [combine_samples.py](combined_samples.py) makes it easy to combine multiple simulation runs into a single file as
a preparation for training.
- [train.py](train.py) trains neural networks on the simulated data.
- [test.py](test.py) steers the evaluation of the estimated likelihood ratio on test or calibration data.
- [calibrate.py](calibrate.py) calibrated network predictions based on histograms of the network output.

All scripts can be called with the argument `--help` to show the command line options. In [scripts/](scripts/) we
collect the actual calls we used (on a HPC environment) during this project.

Generally, the simulation code resides in [simulation](simulation/), while the inference code is in the
[inference](inference/) folder. Notebooks in [notebooks](notebooks/) contain the plotting code.


## References

If you use this code, please cite our paper:

```
(TBA)
```

