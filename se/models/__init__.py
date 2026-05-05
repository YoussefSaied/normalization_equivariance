import torch.nn as nn

from se.configs import TrainConfig, resolve_num_channels
from .dncnn import DnCNN
from .fdncnn import FDnCNN
from .restormer import Restormer
from .swinir import SwinIR
from .wrappers import (
    IdemWrapper,
    NormEquivariant,
    NormEquivariantInput,
    ScaleEquivariant,
)


MODEL_REGISTRY = {
    "dncnn": DnCNN,
    "fdncnn": FDnCNN,
    "swinir": SwinIR,
    "restormer": Restormer,
}
WRAPPERS_REGISTRY = {
    "idem": IdemWrapper,
    "scale-equiv": ScaleEquivariant,
    "norm-equiv": NormEquivariant,
    "norm-equiv-input": NormEquivariantInput,
}


def build_model(cfg: TrainConfig) -> nn.Module:
    model_mode = cfg.model_cfg.model_mode
    wrapper_mode = cfg.model_cfg.wrapper_mode
    in_channels = resolve_num_channels(cfg)

    # if model_mode != "ordinary":
    #     assert (
    #         wrapper_mode == "idem"
    #     ), "For equivariant models, wrapper mode must be 'idem'"
    # if wrapper_mode != "idem":
    #     assert (
    #         model_mode == "ordinary"
    #     ), "For wrapped models, model mode must be 'ordinary'"

    model = MODEL_REGISTRY[cfg.model].build_model(
        cfg.model_cfg, in_channels=in_channels
    )
    wrapper = WRAPPERS_REGISTRY[wrapper_mode]
    return wrapper(model, pred_mode=cfg.model_cfg.pred_mode)
