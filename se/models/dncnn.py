import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from se.models.blocks import BFBatchNorm2d
from se.configs import ModelConfig


class DnCNN(nn.Module):
    """DnCNN as defined in https://arxiv.org/abs/1608.03981
    reference implementation: https://github.com/SaoYan/DnCNN-PyTorch"""

    def __init__(
        self,
        in_channels=1,
        depth=20,
        hidden_size=64,
        kernel_size=3,
        bias=False,
        no_bn=False,  # becomes FDnCNN if True
        model_mode="ordinary",
    ):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        assert model_mode in [
            "ordinary",
            "scale-equiv",
        ], "Only ordinary and scale-equiv modes are implemented"
        self.model_mode = model_mode
        scale_equivariance = self.model_mode == "scale-equiv"
        self.bias = bias or not scale_equivariance
        self.no_bn = no_bn
        if not bias:
            norm_layer = BFBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.depth = depth

        self.first_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            bias=self.bias,
        )

        hidden_layer_list_: list[nn.Module] = []  # (self.depth - 2)
        bn_layer_list_: list[nn.Module] = []  # (self.depth - 2)

        for i in range(self.depth - 2):
            conv = nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=padding,
                bias=self.bias,
            )
            hidden_layer_list_.append(conv)
            if not self.no_bn:
                bn_layer_list_.append(norm_layer(hidden_size))
            else:
                bn_layer_list_.append(nn.Identity())

        self.hidden_layer_list = nn.ModuleList(hidden_layer_list_)
        self.bn_layer_list = nn.ModuleList(bn_layer_list_)
        self.last_layer = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=self.bias,
        )

        self._initialize_weights()

    @classmethod
    def build_model(cls, model_cfg: ModelConfig, in_channels: int = 1):
        bias = model_cfg.model_mode == "ordinary"
        return cls(
            in_channels=in_channels,
            hidden_size=64,
            depth=20,
            no_bn=False,
            bias=bias,
            model_mode=model_cfg.model_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x

        out = self.first_layer(x)
        out = F.relu(out)

        for i in range(self.depth - 2):
            out = self.hidden_layer_list[i](out)
            out = self.bn_layer_list[i](out)
            out = F.relu(out)

        out = self.last_layer(out)

        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, BFBatchNorm2d):
                m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clamp_(
                    -0.025, 0.025
                )
                init.constant_(m.bias, 0)
