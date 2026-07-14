"""Deployment-friendly dense BEV adapters."""

import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVDownsample(nn.Sequential):
    """Downsample a dense pillar BEV before the shared SECOND backbone."""

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 norm_eps=1e-3,
                 norm_momentum=0.01,
                 identity_init=True):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False),
            nn.BatchNorm2d(
                out_channels, eps=norm_eps, momentum=norm_momentum),
            nn.ReLU(inplace=True))
        if identity_init:
            if in_channels != out_channels:
                raise ValueError('identity_init requires matching channels')
            conv, norm = self[0], self[1]
            with torch.no_grad():
                conv.weight.zero_()
                center = kernel_size // 2
                indices = torch.arange(in_channels)
                conv.weight[indices, indices, center, center] = 1
                norm.weight.fill_(1)
                norm.bias.zero_()
