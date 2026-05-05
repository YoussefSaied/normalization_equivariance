# %% imports
import torch
import torch.nn as nn

from se.configs import PredMode


def pred_mode_adapter(
    pred_mode: PredMode, out: torch.Tensor, input: torch.Tensor
) -> torch.Tensor:
    if pred_mode == "residual":
        return input - out
    elif pred_mode == "direct":
        return out
    else:
        raise ValueError(f"Unknown pred_mode: {pred_mode}")


class IdemWrapper(nn.Module):
    def __init__(self, model: nn.Module, pred_mode: PredMode = "residual"):
        super().__init__()
        self.model = model
        self.pred_mode: PredMode = pred_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return pred_mode_adapter(self.pred_mode, out, x)


class ScaleEquivariant(nn.Module):
    def __init__(self, model: nn.Module, pred_mode: PredMode = "residual"):
        super().__init__()
        self.model = model
        self.pred_mode: PredMode = pred_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).pow(0.5)  # avg L2 norm
        model_input = x / (scale + 1e-5)
        out = self.model(model_input)
        out = out * scale
        return pred_mode_adapter(self.pred_mode, out, x)


class NormEquivariant(nn.Module):
    def __init__(self, model: nn.Module, pred_mode: PredMode = "residual"):
        super().__init__()
        self.model = model
        self.pred_mode: PredMode = pred_mode

    def forward(self, x: torch.Tensor, eps=0) -> torch.Tensor:
        scale = x.std(
            dim=(1, 2, 3), keepdim=True
        )  # avg L2 norm; after mean subtraction
        shift = x.mean(dim=(1, 2, 3), keepdim=True)
        model_input = (x - shift) / (scale + eps)
        out = self.model(model_input)

        # pred_mode_adapter
        if self.pred_mode == "residual":
            out = x - out * scale
        else:
            out = out * scale + shift

        return out


class NormEquivariantInput(nn.Module):
    def __init__(self, model: nn.Module, pred_mode: PredMode = "residual"):
        super().__init__()
        self.model = model
        self.pred_mode: PredMode = pred_mode

    def forward(self, x: torch.Tensor, eps=0) -> torch.Tensor:
        # Normalize input with mean/std; do not undo normalization on the output.
        scale = x.std(dim=(1, 2, 3), keepdim=True)
        shift = x.mean(dim=(1, 2, 3), keepdim=True)
        model_input = (x - shift) / (scale + eps)
        out = self.model(model_input)
        return pred_mode_adapter(self.pred_mode, out, x)
