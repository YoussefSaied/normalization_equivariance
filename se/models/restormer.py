"""Restormer architecture adapted for image denoising.

Reference:
https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
"""

import numbers
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from se.configs import ModelConfig


def to_3d(x: torch.Tensor) -> torch.Tensor:
    """Convert BCHW features to BNC (N = H*W) for channel-wise layer norm."""
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert BNC (N = H*W) back to BCHW."""
    b, _, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | Sequence[int]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("normalized_shape must be 1D.")

        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | Sequence[int]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("normalized_shape must be 1D.")

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str = "WithBias") -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError(f"Unknown layer_norm_type: {layer_norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)."""

    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA)."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        head_dim = c // self.num_heads

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    """Restormer for denoising."""

    def __init__(
        self,
        inp_channels: int = 1,
        out_channels: int = 1,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",  # also supports "BiasFree"
        dual_pixel_task: bool = False,  # kept for parity with reference code
    ) -> None:
        super().__init__()

        if len(num_blocks) != 4:
            raise ValueError("num_blocks must contain 4 values.")
        if len(heads) != 4:
            raise ValueError("heads must contain 4 values.")

        self.required_multiple = 8  # three downsampling stages (2^3)
        self.dual_pixel_task = dual_pixel_task

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(
                dim, int(dim * 2**1), kernel_size=1, bias=bias
            )

        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    @classmethod
    def build_model(
        cls, model_cfg: ModelConfig, in_channels: int = 1
    ) -> "Restormer":
        assert (
            model_cfg.model_mode == "ordinary"
        ), "Restormer currently supports only ordinary mode."
        return cls(inp_channels=in_channels, out_channels=in_channels)

    def _check_image_size(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.size()
        mod_pad_h = (self.required_multiple - h % self.required_multiple) % (
            self.required_multiple
        )
        mod_pad_w = (self.required_multiple - w % self.required_multiple) % (
            self.required_multiple
        )
        if mod_pad_h == 0 and mod_pad_w == 0:
            return x, h, w
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")
        return x, h, w

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_img, h, w = self._check_image_size(inp_img)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1[:, :, :h, :w]
