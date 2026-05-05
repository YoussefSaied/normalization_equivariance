from dataclasses import dataclass, field
import os
from typing import Literal, Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ModelMode = Literal["ordinary", "scale-equiv", "norm-equiv"]
WrapperMode = Literal["idem", "scale-equiv", "norm-equiv", "norm-equiv-input"]
PredMode = Literal["residual", "direct"]
NoiseType = Literal["gaussian", "laplace", "uniform", "rayleigh"]
ImageMode = Literal["gray", "rgb"]
TrainObjective = Literal["supervised", "n2n"]


@dataclass
class DatasetConfig:
    train_path: str = f"{PROJECT_ROOT}/data"
    train_dataset_type: str = "h"  # "m" | "h"
    image_mode: Optional[ImageMode] = None
    train_image_dirs: list[str] = field(
        default_factory=lambda: ["BSD400", "DIV2K", "Flickr2K", "WaterlooExploration"]
    )  # used if dataset_type == "h"
    valid_path: Optional[list[str]] = None
    valid_max_images: int | None = None
    s_patch_size: int = 70
    s_samples_per_epoch: int | None = None
    batch_size: int = 128
    noise_type: NoiseType = "gaussian"


@dataclass
class WandbConfig:
    project: str = "scale_equivariance"
    name: str = "experiment"
    mode: str = "online"  # "online" | "offline" | "disabled"


@dataclass
class ModelConfig:
    model_mode: ModelMode = "ordinary"  # "ordinary" | "scale-equiv" | "norm-equiv"
    wrapper_mode: WrapperMode = (
        "idem"  # "idem" | "scale-equiv" | "norm-equiv" | "norm-equiv-input"
    )
    pred_mode: PredMode = "residual"  # "residual" | "direct"


@dataclass
class TrainConfig:
    train_path: str = f"{PROJECT_ROOT}/data"  # m dataset path
    test_path: list[str] = field(
        default_factory=lambda: [
            f"{PROJECT_ROOT}/data/Set12",
            # f"{PROJECT_ROOT}/data/Set68",
        ]
    )
    train_dataset_type: str = "m"  # "m" -> H5 patches, "h" -> on-the-fly patches
    image_mode: Optional[ImageMode] = None

    # h specific
    train_image_dirs: list[str] = field(
        default_factory=lambda: ["BSD400", "DIV2K", "Flickr2K", "WaterlooExploration"]
    )  # directories for "h" dataset
    valid_path: Optional[list[str]] = None
    valid_max_images: int | None = None
    s_patch_size: int = 70
    s_steps_per_epoch: int = 3000
    s_samples_per_epoch: int | None = None

    loss_type: str = "k"  # "k" | "l1" | "l2"
    batch_size: int = 128  # originally 128
    num_epochs: int = 100
    num_steps: int | None = int(5e5)  # m is 3e5
    psnr_eval_sigma_values: Optional[list[float]] = None  # optional override for eval sweep
    model: str = "dncnn"
    train_objective: TrainObjective = "supervised"
    min_noise: float = 50.0
    max_noise: float = 50.0
    noise_type: NoiseType = "gaussian"
    soft_ne_loss: bool = False
    soft_ne_alpha_min: float = 0.0
    soft_ne_alpha_max: float = 1.0
    soft_ne_mu_min: float = 0.0
    soft_ne_mu_max: float = 1.0
    lr: float = 1e-4  # m is 1e-3
    lr_halving_epochs: list[int] | None = field(
        default_factory=lambda: [50, 60, 70, 80, 90, 100]
    )
    lr_halving_steps: int | None = None
    valid_interval: int = 10
    log_interval: int = 100
    seed: int = 0
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    wandb_cfg: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        if self.s_samples_per_epoch is None:
            self.s_samples_per_epoch = self.s_steps_per_epoch * self.batch_size


def resolve_image_mode(cfg: TrainConfig | DatasetConfig) -> str:
    image_mode = getattr(cfg, "image_mode", None)
    if image_mode is not None:
        return image_mode.lower()
    return cfg.train_dataset_type.lower()


def resolve_num_channels(cfg: TrainConfig | DatasetConfig) -> int:
    return 3 if resolve_image_mode(cfg) == "rgb" else 1
