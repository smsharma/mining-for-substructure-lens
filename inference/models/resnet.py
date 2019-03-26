from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.autograd import grad
import logging

logger = logging.getLogger(__name__)


import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetRatioEstimator(nn.Module):
    def __init__(self, n_parameters, cfg=50, input_mean=None, input_std=None, log_input=False, num_classes=1000, zero_init_residual=False, norm_layer=None):
        super(ResNetRatioEstimator, self).__init__()

        self.input_mean = input_mean
        self.input_std = input_std
        self.log_input = log_input

        block, layers = self._load_cfg(cfg)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion + n_parameters, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, theta, x, track_score=True, return_grad_x=False, create_gradient_graph=True):
        # Track gradients
        if track_score and not theta.requires_grad:
            theta.requires_grad = True
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # Preprocessing
        h = self._preprocess(x)

        # ResNet
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)

        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        h = self.avgpool(h)
        h = h.view(h.size(0), -1)
        h = torch.cat((h, theta), 1)
        log_r = self.fc(h)

        # Transform to outputs
        s = self.sigmoid(-1.0 * log_r)

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

    @staticmethod
    def _load_cfg(cfg):
        if cfg == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif cfg == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif cfg == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif cfg == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif cfg == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
        else:
            raise ValueError("Unknown ResNet configuration {}, use 18, 34, 50, 101, or 152!".format(cfg))

        return block, layers

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _preprocess(self, x):
        if self.log_input:
            x = torch.log(1.0 + x)
        if self.input_mean is not None and self.input_std is not None:
            x = x - self.input_mean
            x = x / self.input_std
        x = x.unsqueeze(1)
        return x
