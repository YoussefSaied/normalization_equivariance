#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

from se.configs import ModelConfig
from .blocks import *

# Code inspired from https://github.com/cszn/DPIR/blob/master/models/network_dncnn.py


class FDnCNN(nn.Module):
    def __init__(
        self, in_channels=1, hidden_size=64, depth=20, blind=True, mode="ordinary"
    ):
        """
        in_channels: channel number of input
        out_nc: channel number of output
        hidden_size: channel number
        nb: total number of conv layers
        """
        super().__init__()

        bias = mode == "ordinary"
        self.blind = blind
        assert blind is True, "non-blind FDnCNN is not implemented yet"

        layers = []
        layers.append(
            conv2d(
                in_channels,
                hidden_size,
                3,
                padding=1,
                bias=bias,
                blind=blind,
                mode=mode,
            )
        )
        layers.append(activation(mode))
        for _ in range(depth - 2):
            layers.append(
                conv2d(hidden_size, hidden_size, 3, padding=1, bias=bias, mode=mode)
            )
            layers.append(activation(mode))
        layers.append(
            conv2d(hidden_size, in_channels, 3, padding=1, bias=False, mode=mode)
        )
        self.fdncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.fdncnn(x)

    @classmethod
    def build_model(cls, model_cfg: ModelConfig, in_channels: int = 1):
        return cls(
            in_channels=in_channels,
            hidden_size=64,
            depth=20,
            mode=model_cfg.model_mode,
        )
