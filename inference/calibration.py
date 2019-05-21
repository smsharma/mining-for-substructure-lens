from __future__ import absolute_import, division, print_function

import logging
import numpy as np

logger = logging.getLogger(__name__)


class HistogramCalibrator:
    def __init__(self, data_num, data_den, mode="dynamic", nbins=100, histrange=None):
        self.range, self.edges = self._find_binning(data_num, data_den, mode, nbins, histrange)

        logger.debug("Setting up histogram")
        logger.debug("  Num: mean %s, std %s, min %s, max %s", np.mean(data_num), np.std(data_num), np.min(data_num), np.max(data_num))
        logger.debug("  Den: mean %s, std %s, min %s, max %s", np.mean(data_den), np.std(data_den), np.min(data_den), np.max(data_den))
        logger.debug("  Binning: %s", self.edges)

        self.hist_num = self._fill_histogram(data_num)
        self.hist_den = self._fill_histogram(data_den)

    def log_likelihood_ratio(self, data):
        indices = self._find_bins(data)
        num = self.hist_num[indices]
        den = self.hist_den[indices]

        llr = np.log(num) - np.log(den)
        return llr

    @staticmethod
    def _find_binning(data_num, data_den, mode, nbins, histrange):
        data = np.hstack((data_num, data_den)).flatten()
        if histrange is None:
            hmin = np.min(data)
            hmax = np.max(data)
        else:
            hmin, hmax = histrange

        if mode == "fixed":
            edges = np.linspace(hmin, hmax, nbins + 1)
        elif mode == "dynamic":
            percentages = 100.0 * np.linspace(0.0, 1.0, nbins)
            edges = np.percentile(data, percentages)
        else:
            raise RuntimeError("Unknown mode {}".format(mode))

        return (hmin, hmax), edges

    def _fill_histogram(self, data, epsilon=1.0e-9):
        histo, _ = np.histogram(data, self.edges, self.range)
        histo = histo / np.sum(histo)
        histo += epsilon
        histo = histo / np.sum(histo)
        return histo

    def _find_bins(self, data):
        indices = np.searchsorted(self.edges, data)
        indices = np.clip(indices - 1, 0, len(self.edges) - 2)
        return indices
