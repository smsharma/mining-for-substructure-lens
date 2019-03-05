from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.autograd import grad
from inference.utils import get_activation_function
import logging

logger = logging.getLogger(__name__)


class DenseRatioEstimator(nn.Module):
    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh"):

        super(DenseRatioEstimator, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables + n_parameters

        # Hidden layers
        for n_hidden_units in n_hidden:
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        self.layers.append(nn.Linear(n_last, 1))

    def forward(
        self,
        theta,
        x,
        track_score=True,
        return_grad_x=False,
        create_gradient_graph=True,
    ):

        """ Calculates estimated log likelihood ratio and the derived score. """

        # Track gradient wrt theta
        if track_score and not theta.requires_grad:
            theta.requires_grad = True

        # Track gradient wrt x
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # log r estimator
        log_r_hat = torch.cat((theta, x), 1)

        for i, layer in enumerate(self.layers):
            if i > 0:
                log_r_hat = self.activation(log_r_hat)
            log_r_hat = layer(log_r_hat)

        # Bayes-optimal s
        s_hat = 1.0 / (1.0 + torch.exp(log_r_hat))

        # Score t
        if track_score:
            t_hat, = grad(
                log_r_hat,
                theta,
                grad_outputs=torch.ones_like(log_r_hat.data),
                # grad_outputs=log_r_hat.data.new(log_r_hat.shape).fill_(1),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
        else:
            t_hat = None

        # Calculate gradient wrt x
        if return_grad_x:
            x_gradient, = grad(
                log_r_hat,
                x,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )

            return s_hat, log_r_hat, t_hat, x_gradient

        return s_hat, log_r_hat, t_hat

    def to(self, *args, **kwargs):
        self = super(DenseRatioEstimator, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self


class Conv2DRatioEstimator(nn.Module):
    def __init__(
        self,
        n_parameters,
        resolution,
        n_conv_layers=3,
        n_dense_layers=3,
        n_feature_maps=128,
        kernel_size=5,
        pooling_size=2,
        n_hidden_dense=128,
        activation="relu",
    ):

        super(Conv2DRatioEstimator, self).__init__()

        assert n_conv_layers > 0

        # Save input
        self.resolution = resolution
        self.n_conv_layers = n_conv_layers
        self.n_dense_layers = n_dense_layers
        self.n_feature_maps = n_feature_maps
        self.kernel_size = kernel_size

        # Build network
        self.conv_layers = nn.ModuleList()
        self.dense_layers = nn.ModuleList()

        last_channels = 1 + n_parameters
        current_size = resolution

        # Convolutional and pooling layers
        for i_conv in range(n_conv_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=n_feature_maps,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=int((kernel_size - 1) / 2),
                    bias=(i_conv > 0),
                )
            )
            self.conv_layers.append(nn.BatchNorm2d(n_feature_maps))
            self.conv_layers.append(get_activation_function(activation))
            self.conv_layers.append(
                nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size)
            )

            last_channels = n_feature_maps
            current_size = current_size // pooling_size

        n_units = int(last_channels * current_size * current_size + n_parameters)

        # Fully connected layers
        for i_conv in range(n_dense_layers - 1):
            self.dense_layers.append(nn.Linear(n_units, n_hidden_dense))
            self.dense_layers.append(get_activation_function(activation))

            n_units = n_hidden_dense

        # Log r layer
        self.dense_layers.append(nn.Linear(n_units, 1))

    def forward(
        self,
        theta,
        x,
        track_score=True,
        return_grad_x=False,
        create_gradient_graph=True,
    ):
        # Track gradients
        if track_score and not theta.requires_grad:
            theta.requires_grad = True
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # Use theta as color layers
        h1 = x.unsqueeze(1)  # (n_batch, 1, res_x, res_y)
        h_theta = theta.unsqueeze(2).unsqueeze(3)  # (n_batch, n_parameters, 1, 1)
        h_theta = h_theta * torch.ones_like(h1)  # (n_batch, n_parameters, res_x, res_y)
        h = torch.cat((h1, h_theta), 1)

        # Convolutional and pooling layers
        for conv_layer in self.conv_layers:
            h = conv_layer(h)

        # Transition to dense layers, add theta again
        h = h.reshape(h.size(0), -1)
        h = torch.cat((h, theta), 1)

        # Dense layers
        for dense_layer in self.dense_layers:
            h = dense_layer(h)

        # Transform to outputs
        log_r = h
        s = 1.0 / (1.0 + torch.exp(log_r))

        # Score and gradient wrt x
        if track_score:
            t, = grad(
                log_r,
                theta,
                grad_outputs=torch.ones_like(log_r.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
        else:
            t = None
        if return_grad_x:
            x_gradient, = grad(
                log_r,
                x,
                grad_outputs=torch.ones_like(log_r.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
        else:
            x_gradient = None

        return s, log_r, t, x_gradient

    def to(self, *args, **kwargs):
        self = super(Conv2DRatioEstimator, self).to(*args, **kwargs)

        for i, layer in enumerate(self.conv_layers):
            self.conv_layers[i] = layer.to(*args, **kwargs)
        for i, layer in enumerate(self.dense_layers):
            self.dense_layers[i] = layer.to(*args, **kwargs)

        return self


class VGG11RatioEstimator(nn.Module):

    def __init__(self, cfg="A", batch_norm=False, init_weights=True):
        super(VGG11RatioEstimator, self).__init__()
        self.features = self._make_layers(cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 + 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
        )
        if init_weights:
            self._initialize_weights()

    def forward(
        self,
        theta,
        x,
        track_score=True,
        return_grad_x=False,
        create_gradient_graph=True,
    ):
        # Track gradients
        if track_score and not theta.requires_grad:
            theta.requires_grad = True
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # VGG11
        h = x.unsqueeze(1)  # (n_batch, 1, res_x, res_y)
        h = self.features(h)
        h = self.avgpool(h)
        h = h.view(h.size(0), -1)
        h = torch.cat((h, theta), 1)
        h = self.classifier(h)

        # Transform to outputs
        log_r = h
        s = 1.0 / (1.0 + torch.exp(log_r))

        # Score and gradient wrt x
        if track_score:
            t, = grad(
                log_r,
                theta,
                grad_outputs=torch.ones_like(log_r.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
        else:
            t = None
        if return_grad_x:
            x_gradient, = grad(
                log_r,
                x,
                grad_outputs=torch.ones_like(log_r.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
        else:
            x_gradient = None

        return s, log_r, t, x_gradient

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layers(cfg='A', batch_norm=False):
        configs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
        }

        layers = []
        in_channels = 1
        for v in configs[cfg]:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
