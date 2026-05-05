import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

# Architectural scale-equivariant and normalization-equivariant operators used
# for the FDnCNN references follow Mohan et al. (2020) and Herbreteau et al.
# (2023).


class BFBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, use_bias=False, affine=True
    ):
        super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)
        self.use_bias = use_bias

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        y = input.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(input.size(1), -1)

        # Per-channel stats over the flattened spatial/batch axes
        mu = y.mean(dim=1) if self.use_bias else None
        var = y.var(dim=1)

        if self.training:
            if self.track_running_stats:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mu  # type: ignore[arg-type]
                    self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var  # type: ignore[arg-type]

            center = mu if self.use_bias else None
            scale_var = var
        else:
            center = self.running_mean if self.use_bias else None
            scale_var = self.running_var

        if center is not None:
            y = y - center.view(-1, 1)

        y = y / (torch.sqrt(scale_var).view(-1, 1) + self.eps)  # type: ignore[arg-type]

        if self.affine:
            y = self.weight.view(-1, 1) * y
            if self.use_bias:
                y = y + self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0, 1)


class AffineConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "reflect",
        blind=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=False,
        )
        self.blind = blind

    def affine(self, w):
        """returns new kernels that encode affine combinations"""
        return (
            w.view(self.out_channels, -1).roll(1, 1).view(w.size())
            - w
            + 1 / w[0, ...].numel()
        )

    def forward(self, input):
        kernel = (
            self.affine(self.weight)
            if self.blind
            else torch.cat(
                (self.affine(self.weight[:, :-1, :, :]), self.weight[:, -1:, :, :]),
                dim=1,
            )
        )
        padding = tuple(
            int(elt) for elt in reversed(self.padding) for _ in range(2)
        )  # used to translate padding arg used by Conv module to the ones used by F.pad
        padding_mode = (
            self.padding_mode if self.padding_mode != "zeros" else "constant"
        )  # used to translate padding_mode arg used by Conv module to the ones used by F.pad
        return F.conv2d(
            F.pad(input, padding, mode=padding_mode),
            kernel,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )


class AffineConvTranspose2d(nn.Module):
    """Affine ConvTranspose2d with kernel=2 and stride=2, implemented using PixelShuffle"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = AffineConv2d(in_channels, 4 * out_channels, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.conv1x1(x), 2)


class SortPool(nn.Module):
    """Channel-wise sort pooling, C must be an even number"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # A trick with relu is used because the derivative for torch.aminmax is not yet implemented and torch.sort is slow.
        N, C, H, W = x.size()
        x1, x2 = torch.split(x.view(N, C // 2, 2, H, W), 1, dim=2)
        diff = F.relu(x1 - x2, inplace=True)
        return torch.cat((x1 - diff, x2 + diff), dim=2).view(N, C, H, W)


class ResidualConnection(nn.Module):
    """Residual connection"""

    def __init__(self, mode="ordinary"):
        super().__init__()

        self.mode = mode
        if mode == "norm-equiv":
            self.alpha = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, x, y):
        if self.mode == "norm-equiv":
            return self.alpha * x + (1 - self.alpha) * y
        return x + y


def conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    blind=True,
    mode="ordinary",
):
    if mode == "ordinary" or mode == "scale-equiv":
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias if mode == "ordinary" else False,
            padding_mode=padding_mode,
        )
    elif mode == "norm-equiv":
        return AffineConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode="reflect",
            blind=blind,
        )
    else:
        raise NotImplementedError(
            "Only ordinary, scale-equiv and norm-equiv modes are implemented"
        )


def upscale2(in_channels, out_channels, bias=True, mode="ordinary"):
    """Upscaling using convtranspose with kernel 2x2 and stride 2"""
    if mode == "ordinary" or mode == "scale-equiv":
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=bias if mode == "ordinary" else False,
        )
    elif mode == "norm-equiv":
        return AffineConvTranspose2d(in_channels, out_channels)
    else:
        raise NotImplementedError(
            "Only ordinary, scale-equiv and norm-equiv modes are implemented"
        )


def activation(mode="ordinary"):
    if mode == "ordinary" or mode == "scale-equiv":
        return nn.ReLU(inplace=True)
    elif mode == "norm-equiv":
        return SortPool()
    else:
        raise NotImplementedError(
            "Only ordinary, scale-equiv and norm-equiv modes are implemented"
        )


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=False, mode="ordinary"):
        super().__init__()

        self.m_res = nn.Sequential(
            conv2d(
                in_channels, in_channels, 3, stride=1, padding=1, bias=bias, mode=mode
            ),
            activation(mode),
            conv2d(
                in_channels, out_channels, 3, stride=1, padding=1, bias=bias, mode=mode
            ),
        )

        self.sum = ResidualConnection(mode)

    def forward(self, x):
        return self.sum(x, self.m_res(x))
