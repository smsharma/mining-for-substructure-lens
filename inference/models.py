from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.autograd import grad
import logging

logger = logging.getLogger(__name__)


class VGG11RatioEstimator(nn.Module):
    def __init__(
        self, n_parameters, cfg="A", input_mean=None, input_std=None, log_input=False, batch_norm=True, init_weights=True
    ):
        super(VGG11RatioEstimator, self).__init__()

        self.input_mean = input_mean
        self.input_std = input_std
        self.log_input = log_input

        self.features = self._make_layers(cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 + n_parameters, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 1)
        )
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def forward(self, theta, x, track_score=True, return_grad_x=False, create_gradient_graph=True):
        # Track gradients
        if track_score and not theta.requires_grad:
            theta.requires_grad = True
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # Preprocessing
        h = self._preprocess(x)

        # VGG11
        h = self.features(h)
        h = self.avgpool(h)
        h = h.view(h.size(0), -1)
        h = torch.cat((h, theta), 1)
        h = self.classifier(h)

        # Transform to outputs
        log_r = h
        # s = 1.0 / (1.0 + torch.exp(log_r))
        s = self.sigmoid(-1. * log_r)
        logger.debug("After r-to-s trafo: %s", s)

        # Score and gradient wrt x
        if track_score:
            t, = grad(log_r, theta, grad_outputs=torch.ones_like(log_r.data), only_inputs=True, create_graph=create_gradient_graph)
        else:
            t = None
        if return_grad_x:
            x_gradient, = grad(log_r, x, grad_outputs=torch.ones_like(log_r.data), only_inputs=True, create_graph=create_gradient_graph)
        else:
            x_gradient = None

        return s, log_r, t, x_gradient

    def _preprocess(self, x):
        if self.log_input:
            x = torch.log(1.0 + x)
        if self.input_mean is not None and self.input_std is not None:
            x = (x - self.input_mean)
            x = x / max(1.0e-6, self.input_std)
        x = x.unsqueeze(1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layers(cfg="A", batch_norm=False):
        configs = {
            "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }

        layers = []
        in_channels = 1
        for v in configs[cfg]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
