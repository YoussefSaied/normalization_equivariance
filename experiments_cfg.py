from copy import deepcopy
from se.configs import TrainConfig, ModelConfig, PROJECT_ROOT


def _set_s_samples_per_epoch(cfg: TrainConfig) -> TrainConfig:
    # Keep steps/epoch stable if batch_size changes after init
    cfg.s_samples_per_epoch = cfg.s_steps_per_epoch * cfg.batch_size
    return cfg


def _with_noise(cfg: TrainConfig, sigma_8bit: float, noise_type: str) -> TrainConfig:
    cfg.min_noise = sigma_8bit
    cfg.max_noise = sigma_8bit
    cfg.noise_type = noise_type  # type: ignore[attr-defined]
    return cfg


def _with_noise_range(
    cfg: TrainConfig, min_noise_8bit: float, max_noise_8bit: float
) -> TrainConfig:
    cfg.min_noise = min_noise_8bit
    cfg.max_noise = max_noise_8bit
    return cfg


def _sigma_unit_to_8bit(sigma: float) -> float:
    return 255.0 * sigma


def _with_eval_sigma_values(
    cfg: TrainConfig, sigma_values: list[float]
) -> TrainConfig:
    cfg.psnr_eval_sigma_values = list(sigma_values)
    return cfg


def _with_train_objective(cfg: TrainConfig, objective: str) -> TrainConfig:
    cfg.train_objective = objective  # type: ignore[assignment]
    return cfg


def _with_color_data(cfg: TrainConfig) -> TrainConfig:
    cfg.image_mode = "rgb"
    cfg.train_image_dirs = ["DIV2K", "Flickr2K", "WaterlooExploration"]
    cfg.valid_path = [f"{PROJECT_ROOT}/data/CBSD68"]
    cfg.valid_max_images = 12
    cfg.test_path = [f"{PROJECT_ROOT}/data/CBSD68"]
    return cfg


def _make_backbone_cfg(
    base_cfg: TrainConfig,
    *,
    model: str,
    wrapper_mode: str,
    patch_size: int,
    batch_size: int,
    lr_halving_steps: int | None = None,
) -> TrainConfig:
    cfg = deepcopy(base_cfg)
    cfg.model = model
    cfg.model_cfg = ModelConfig(
        model_mode="ordinary",
        pred_mode="direct",
        wrapper_mode=wrapper_mode,  # type: ignore[arg-type]
    )
    cfg.lr_halving_steps = lr_halving_steps
    cfg.s_patch_size = patch_size
    cfg.batch_size = batch_size
    return _set_s_samples_per_epoch(cfg)


# %% Gaussian dncnn experiments
cfg = TrainConfig(
    train_dataset_type="h",
    test_path=[f"{PROJECT_ROOT}/data/Set12"],
    loss_type="l2",
    lr=1e-4,
    lr_halving_epochs=None,
    lr_halving_steps=None,
    s_patch_size=70,
)

# 50 noise level
cfg_50 = deepcopy(cfg)
cfg_50.min_noise = 50.0
cfg_50.max_noise = 50.0

## 50 noise level, FDnCNN model, NE
cfg_50_fdncnn_ne = deepcopy(cfg_50)
cfg_50_fdncnn_ne.model = "fdncnn"
cfg_50_fdncnn_ne.num_steps = 900_000
cfg_50_fdncnn_ne.model_cfg = ModelConfig(
    model_mode="norm-equiv", pred_mode="direct", wrapper_mode="idem"
)

## 50 noise level, FDnCNN model, WNE
cfg_50_fdncnn_wne = deepcopy(cfg_50)
cfg_50_fdncnn_wne.model = "fdncnn"
cfg_50_fdncnn_wne.num_steps = 900_000
cfg_50_fdncnn_wne.lr_halving_steps = int(1e5)
cfg_50_fdncnn_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)

## 50 noise level, FDnCNN model, SE
cfg_50_fdncnn_se = deepcopy(cfg_50)
cfg_50_fdncnn_se.model = "fdncnn"
cfg_50_fdncnn_se.model_cfg = ModelConfig(
    model_mode="scale-equiv", pred_mode="direct", wrapper_mode="idem"
)
cfg_50_fdncnn_se.loss_type = "l1"

## 50 noise level, FDnCNN model, O
cfg_50_fdncnn_o = deepcopy(cfg_50)
cfg_50_fdncnn_o.model = "fdncnn"
cfg_50_fdncnn_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_50_fdncnn_o.loss_type = "l1"

## 50 noise level, DnCNN model, WNE
cfg_50_dncnn_wne = deepcopy(cfg_50)
cfg_50_dncnn_wne.model = "dncnn"
cfg_50_dncnn_wne.lr_halving_steps = int(1e5)
cfg_50_dncnn_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)

## 50 noise level, DnCNN model, O
cfg_50_dncnn_o = deepcopy(cfg_50)
cfg_50_dncnn_o.model = "dncnn"
cfg_50_dncnn_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_50_dncnn_o.loss_type = "l1"

# 25 noise level
cfg_25 = deepcopy(cfg)
cfg_25.min_noise = 25.0
cfg_25.max_noise = 25.0

## 25 noise level, FDnCNN model, NE
cfg_25_fdncnn_ne = deepcopy(cfg_25)
cfg_25_fdncnn_ne.model = "fdncnn"
cfg_25_fdncnn_ne.num_steps = 900_000
cfg_25_fdncnn_ne.model_cfg = ModelConfig(
    model_mode="norm-equiv", pred_mode="direct", wrapper_mode="idem"
)
## 25 noise level, FDnCNN model, WNE
cfg_25_fdncnn_wne = deepcopy(cfg_25)
cfg_25_fdncnn_wne.model = "fdncnn"
cfg_25_fdncnn_wne.num_steps = 900_000
cfg_25_fdncnn_wne.lr_halving_steps = int(1e5)
cfg_25_fdncnn_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)

## 25 noise level, FDnCNN model, SE
cfg_25_fdncnn_se = deepcopy(cfg_25)
cfg_25_fdncnn_se.model = "fdncnn"
cfg_25_fdncnn_se.model_cfg = ModelConfig(
    model_mode="scale-equiv", pred_mode="direct", wrapper_mode="idem"
)
cfg_25_fdncnn_se.loss_type = "l1"

## 25 noise level, FDnCNN model, O
cfg_25_fdncnn_o = deepcopy(cfg_25)
cfg_25_fdncnn_o.model = "fdncnn"
cfg_25_fdncnn_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_25_fdncnn_o.loss_type = "l1"

## 25 noise level, DnCNN model, WNE
cfg_25_dncnn_wne = deepcopy(cfg_25)
cfg_25_dncnn_wne.model = "dncnn"
cfg_25_dncnn_wne.lr_halving_steps = int(1e5)
cfg_25_dncnn_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)

## 25 noise level, DnCNN model, O
cfg_25_dncnn_o = deepcopy(cfg_25)
cfg_25_dncnn_o.model = "dncnn"
cfg_25_dncnn_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_25_dncnn_o.loss_type = "l1"

# 10 noise level
cfg_10 = deepcopy(cfg)
cfg_10.min_noise = 10.0
cfg_10.max_noise = 10.0

# 0.1 noise level (8-bit units)
cfg_0p1 = deepcopy(cfg)
cfg_0p1.min_noise = 0.1
cfg_0p1.max_noise = 0.1

## 10 noise level, FDnCNN model, NE
cfg_10_fdncnn_ne = deepcopy(cfg_10)
cfg_10_fdncnn_ne.model = "fdncnn"
cfg_10_fdncnn_ne.num_steps = 900_000
cfg_10_fdncnn_ne.model_cfg = ModelConfig(
    model_mode="norm-equiv", pred_mode="direct", wrapper_mode="idem"
)
## 10 noise level, FDnCNN model, WNE
cfg_10_fdncnn_wne = deepcopy(cfg_10)
cfg_10_fdncnn_wne.model = "fdncnn"
cfg_10_fdncnn_wne.num_steps = 900_000
cfg_10_fdncnn_wne.lr_halving_steps = int(1e5)
cfg_10_fdncnn_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)

## 10 noise level, FDnCNN model, SE
cfg_10_fdncnn_se = deepcopy(cfg_10)
cfg_10_fdncnn_se.model = "fdncnn"
cfg_10_fdncnn_se.model_cfg = ModelConfig(
    model_mode="scale-equiv", pred_mode="direct", wrapper_mode="idem"
)
cfg_10_fdncnn_se.loss_type = "l1"

## 10 noise level, FDnCNN model, O
cfg_10_fdncnn_o = deepcopy(cfg_10)
cfg_10_fdncnn_o.model = "fdncnn"
cfg_10_fdncnn_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_10_fdncnn_o.loss_type = "l1"

## 10 noise level, DnCNN model, WNE
cfg_10_dncnn_wne = deepcopy(cfg_10)
cfg_10_dncnn_wne.model = "dncnn"
cfg_10_dncnn_wne.lr_halving_steps = int(1e5)
cfg_10_dncnn_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)

## 10 noise level, DnCNN model, WNEI (input-only NE wrapper)
cfg_10_dncnn_wnei = deepcopy(cfg_10)
cfg_10_dncnn_wnei.model = "dncnn"
cfg_10_dncnn_wnei.lr_halving_steps = int(1e5)
cfg_10_dncnn_wnei.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv-input"
)

## 10 noise level, DnCNN model, O
cfg_10_dncnn_o = deepcopy(cfg_10)
cfg_10_dncnn_o.model = "dncnn"
cfg_10_dncnn_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_10_dncnn_o.loss_type = "l1"

# %% non-Gaussian experiments
# Input PSNR -> sigma (in [0,1]) -> sigma_8bit:
# 22 dB -> ~0.0794 -> ~20.2
# 25 dB -> ~0.0562 -> ~14.3
# 17 dB (Rayleigh, nonzero mean) -> scale down by sqrt(1 + 1.913^2) ≈ sqrt(4.659)
#     target sigma ≈ 0.0655 -> sigma_8bit ≈ 16.7

sigma22_laplace = 20
sigma25_uniform = 14
sigma17_rayleigh = 17  # approx 16.7 rounded to nearest integer

# Base configs for non-Gaussian experiments
cfg_laplace_22 = _with_noise(deepcopy(cfg), sigma22_laplace, "laplace")
cfg_uniform_25 = _with_noise(deepcopy(cfg), sigma25_uniform, "uniform")
cfg_rayleigh_17 = _with_noise(deepcopy(cfg), sigma17_rayleigh, "rayleigh")
cfg_rayleigh_17.psnr_eval_sigma_values = [
    s / 255.0 for s in (1, 2, 3, 5, 8, 12, 17, 25, 36, 50, 70, 95)
]

# Laplace @22 dB
cfg_laplace_22_dncnn_o = deepcopy(cfg_laplace_22)
cfg_laplace_22_dncnn_o.model = "dncnn"
cfg_laplace_22_dncnn_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_laplace_22_dncnn_o.loss_type = "l1"

cfg_laplace_22_dncnn_wne = deepcopy(cfg_laplace_22_dncnn_o)
cfg_laplace_22_dncnn_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_laplace_22_dncnn_wne.loss_type = "l1"

cfg_laplace_22_fdncnn_ne = deepcopy(cfg_laplace_22_dncnn_o)
cfg_laplace_22_fdncnn_ne.model = "fdncnn"
cfg_laplace_22_fdncnn_ne.num_steps = 900_000
cfg_laplace_22_fdncnn_ne.model_cfg = ModelConfig(
    model_mode="norm-equiv", pred_mode="direct", wrapper_mode="idem"
)
cfg_laplace_22_fdncnn_ne.loss_type = "l1"

cfg_laplace_22_fdncnn_se = deepcopy(cfg_laplace_22_dncnn_o)
cfg_laplace_22_fdncnn_se.model = "fdncnn"
cfg_laplace_22_fdncnn_se.model_cfg = ModelConfig(
    model_mode="scale-equiv", pred_mode="direct", wrapper_mode="idem"
)

# Uniform @25 dB
cfg_uniform_25_dncnn_o = _with_noise(deepcopy(cfg), sigma25_uniform, "uniform")
cfg_uniform_25_dncnn_o.model = "dncnn"
cfg_uniform_25_dncnn_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_uniform_25_dncnn_o.loss_type = "l1"

cfg_uniform_25_dncnn_wne = deepcopy(cfg_uniform_25_dncnn_o)
cfg_uniform_25_dncnn_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_uniform_25_dncnn_wne.loss_type = "l1"

cfg_uniform_25_fdncnn_ne = deepcopy(cfg_uniform_25_dncnn_o)
cfg_uniform_25_fdncnn_ne.model = "fdncnn"
cfg_uniform_25_fdncnn_ne.num_steps = 900_000
cfg_uniform_25_fdncnn_ne.model_cfg = ModelConfig(
    model_mode="norm-equiv", pred_mode="direct", wrapper_mode="idem"
)
cfg_uniform_25_fdncnn_ne.loss_type = "l1"

cfg_uniform_25_fdncnn_se = deepcopy(cfg_uniform_25_dncnn_o)
cfg_uniform_25_fdncnn_se.model = "fdncnn"
cfg_uniform_25_fdncnn_se.model_cfg = ModelConfig(
    model_mode="scale-equiv", pred_mode="direct", wrapper_mode="idem"
)

# Rayleigh @17 dB
cfg_rayleigh_17_dncnn_o = deepcopy(cfg_rayleigh_17)
cfg_rayleigh_17_dncnn_o.model = "dncnn"
cfg_rayleigh_17_dncnn_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_rayleigh_17_dncnn_o.loss_type = "l1"

cfg_rayleigh_17_dncnn_wne = deepcopy(cfg_rayleigh_17_dncnn_o)
cfg_rayleigh_17_dncnn_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_rayleigh_17_dncnn_wne.lr_halving_steps = int(1e5)
cfg_rayleigh_17_dncnn_wne.loss_type = "l1"

cfg_rayleigh_17_fdncnn_ne = deepcopy(cfg_rayleigh_17_dncnn_o)
cfg_rayleigh_17_fdncnn_ne.model = "fdncnn"
cfg_rayleigh_17_fdncnn_ne.num_steps = 900_000
cfg_rayleigh_17_fdncnn_ne.model_cfg = ModelConfig(
    model_mode="norm-equiv", pred_mode="direct", wrapper_mode="idem"
)
cfg_rayleigh_17_fdncnn_ne.loss_type = "l1"

cfg_rayleigh_17_fdncnn_wne_se = deepcopy(cfg_rayleigh_17_dncnn_o)
cfg_rayleigh_17_fdncnn_wne_se.model = "fdncnn"
cfg_rayleigh_17_fdncnn_wne_se.num_steps = 900_000
cfg_rayleigh_17_fdncnn_wne_se.model_cfg = ModelConfig(
    model_mode="scale-equiv", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_rayleigh_17_fdncnn_wne_se.lr_halving_steps = int(1e5)
cfg_rayleigh_17_fdncnn_wne_se.loss_type = "l1"

cfg_rayleigh_17_fdncnn_se = deepcopy(cfg_rayleigh_17_dncnn_o)
cfg_rayleigh_17_fdncnn_se.model = "fdncnn"
cfg_rayleigh_17_fdncnn_se.model_cfg = ModelConfig(
    model_mode="scale-equiv", pred_mode="direct", wrapper_mode="idem"
)

# %% SwinIR experiments

# SwinIR (lite denoising) configs: ordinary (_o) and WNE (_wne)
swinir_patch_size = 64
swinir_batch_size = 32
levac_fig1_sigma_values = [0.01, 0.02, 0.05, 0.075, 0.10, 0.15]
levac_sigma10_sigma_values = [*levac_fig1_sigma_values, 10.0 / 255.0]
levac_sigma25_sigma_values = [*levac_fig1_sigma_values, 25.0 / 255.0]
levac_sigma50_sigma_values = [*levac_fig1_sigma_values, 50.0 / 255.0]


def _make_swinir_levac_family(
    base_cfg: TrainConfig,
) -> tuple[TrainConfig, TrainConfig, TrainConfig, TrainConfig]:
    supervised = _make_backbone_cfg(
        base_cfg,
        model="swinir",
        wrapper_mode="idem",
        patch_size=swinir_patch_size,
        batch_size=swinir_batch_size,
    )

    baseline_n2n = _with_train_objective(deepcopy(supervised), "n2n")

    softne_n2n = deepcopy(baseline_n2n)
    softne_n2n.soft_ne_loss = True
    softne_n2n.soft_ne_alpha_min = 0.0
    softne_n2n.soft_ne_alpha_max = 1.0
    softne_n2n.soft_ne_mu_min = 0.0
    softne_n2n.soft_ne_mu_max = 1.0

    wne_n2n = _make_backbone_cfg(
        base_cfg,
        model="swinir",
        wrapper_mode="norm-equiv",
        patch_size=swinir_patch_size,
        batch_size=swinir_batch_size,
        lr_halving_steps=int(1e5),
    )
    wne_n2n = _with_train_objective(wne_n2n, "n2n")

    return supervised, baseline_n2n, softne_n2n, wne_n2n


cfg_50_swinir_o = deepcopy(cfg_50)
cfg_50_swinir_o.model = "swinir"
cfg_50_swinir_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_50_swinir_o.s_patch_size = swinir_patch_size
cfg_50_swinir_o.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_50_swinir_o)

cfg_25_swinir_o = deepcopy(cfg_25)
cfg_25_swinir_o.model = "swinir"
cfg_25_swinir_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_25_swinir_o.s_patch_size = swinir_patch_size
cfg_25_swinir_o.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_25_swinir_o)

cfg_10_swinir_o = deepcopy(cfg_10)
cfg_10_swinir_o.model = "swinir"
cfg_10_swinir_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_10_swinir_o.s_patch_size = swinir_patch_size
cfg_10_swinir_o.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_10_swinir_o)

cfg_10_swinir_softne = deepcopy(cfg_10_swinir_o)
cfg_10_swinir_softne.soft_ne_loss = True
cfg_10_swinir_softne.soft_ne_alpha_min = 0.0
cfg_10_swinir_softne.soft_ne_alpha_max = 1.0
cfg_10_swinir_softne.soft_ne_mu_min = 0.0
cfg_10_swinir_softne.soft_ne_mu_max = 1.0

cfg_50_swinir_wne = deepcopy(cfg_50)
cfg_50_swinir_wne.model = "swinir"
cfg_50_swinir_wne.lr_halving_steps = int(1e5)
cfg_50_swinir_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_50_swinir_wne.s_patch_size = swinir_patch_size
cfg_50_swinir_wne.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_50_swinir_wne)

cfg_25_swinir_wne = deepcopy(cfg_25)
cfg_25_swinir_wne.model = "swinir"
cfg_25_swinir_wne.lr_halving_steps = int(1e5)
cfg_25_swinir_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_25_swinir_wne.s_patch_size = swinir_patch_size
cfg_25_swinir_wne.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_25_swinir_wne)

cfg_10_swinir_wne = deepcopy(cfg_10)
cfg_10_swinir_wne.model = "swinir"
cfg_10_swinir_wne.lr_halving_steps = int(1e5)
cfg_10_swinir_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_10_swinir_wne.s_patch_size = swinir_patch_size
cfg_10_swinir_wne.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_10_swinir_wne)

cfg_0p1_swinir_o = deepcopy(cfg_0p1)
cfg_0p1_swinir_o.model = "swinir"
cfg_0p1_swinir_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_0p1_swinir_o.s_patch_size = swinir_patch_size
cfg_0p1_swinir_o.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_0p1_swinir_o)

cfg_0p1_swinir_wne = deepcopy(cfg_0p1)
cfg_0p1_swinir_wne.model = "swinir"
cfg_0p1_swinir_wne.lr_halving_steps = int(1e5)
cfg_0p1_swinir_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_0p1_swinir_wne.s_patch_size = swinir_patch_size
cfg_0p1_swinir_wne.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_0p1_swinir_wne)

# %% SwinIR N2N experiments inspired by Levac et al. Fig. 1

cfg_sigma0p05 = _with_eval_sigma_values(
    _with_noise(deepcopy(cfg), _sigma_unit_to_8bit(0.05), "gaussian"),
    levac_fig1_sigma_values,
)
cfg_sigma0p075 = _with_eval_sigma_values(
    _with_noise(deepcopy(cfg), _sigma_unit_to_8bit(0.075), "gaussian"),
    levac_fig1_sigma_values,
)
cfg_sigma0p10 = _with_eval_sigma_values(
    _with_noise(deepcopy(cfg), _sigma_unit_to_8bit(0.10), "gaussian"),
    levac_fig1_sigma_values,
)
cfg_sigma10 = _with_eval_sigma_values(
    _with_noise(deepcopy(cfg), 10.0, "gaussian"),
    levac_sigma10_sigma_values,
)
cfg_sigma25 = _with_eval_sigma_values(
    _with_noise(deepcopy(cfg), 25.0, "gaussian"),
    levac_sigma25_sigma_values,
)
cfg_sigma50 = _with_eval_sigma_values(
    _with_noise(deepcopy(cfg), 50.0, "gaussian"),
    levac_sigma50_sigma_values,
)

(
    cfg_sigma0p05_swinir_sup,
    cfg_sigma0p05_swinir_n2n,
    cfg_sigma0p05_swinir_softne_n2n,
    cfg_sigma0p05_swinir_wne_n2n,
) = _make_swinir_levac_family(cfg_sigma0p05)

(
    cfg_sigma0p075_swinir_sup,
    cfg_sigma0p075_swinir_n2n,
    cfg_sigma0p075_swinir_softne_n2n,
    cfg_sigma0p075_swinir_wne_n2n,
) = _make_swinir_levac_family(cfg_sigma0p075)

(
    cfg_sigma0p10_swinir_sup,
    cfg_sigma0p10_swinir_n2n,
    cfg_sigma0p10_swinir_softne_n2n,
    cfg_sigma0p10_swinir_wne_n2n,
) = _make_swinir_levac_family(cfg_sigma0p10)

(
    cfg_sigma10_swinir_sup,
    cfg_sigma10_swinir_n2n,
    cfg_sigma10_swinir_softne_n2n,
    cfg_sigma10_swinir_wne_n2n,
) = _make_swinir_levac_family(cfg_sigma10)

(
    cfg_sigma25_swinir_sup,
    cfg_sigma25_swinir_n2n,
    cfg_sigma25_swinir_softne_n2n,
    cfg_sigma25_swinir_wne_n2n,
) = _make_swinir_levac_family(cfg_sigma25)

(
    cfg_sigma50_swinir_sup,
    cfg_sigma50_swinir_n2n,
    cfg_sigma50_swinir_softne_n2n,
    cfg_sigma50_swinir_wne_n2n,
) = _make_swinir_levac_family(cfg_sigma50)

# %% SwinIR non-gaussian

# Laplace @22 dB
cfg_laplace_22_swinir_o = deepcopy(cfg_laplace_22)
cfg_laplace_22_swinir_o.model = "swinir"
cfg_laplace_22_swinir_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_laplace_22_swinir_o.s_patch_size = swinir_patch_size
cfg_laplace_22_swinir_o.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_laplace_22_swinir_o)

cfg_laplace_22_swinir_wne = deepcopy(cfg_laplace_22_swinir_o)
cfg_laplace_22_swinir_wne.lr_halving_steps = int(1e5)
cfg_laplace_22_swinir_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
_set_s_samples_per_epoch(cfg_laplace_22_swinir_wne)

# Uniform @25 dB
cfg_uniform_25_swinir_o = deepcopy(cfg_uniform_25)
cfg_uniform_25_swinir_o.model = "swinir"
cfg_uniform_25_swinir_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_uniform_25_swinir_o.s_patch_size = swinir_patch_size
cfg_uniform_25_swinir_o.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_uniform_25_swinir_o)

cfg_uniform_25_swinir_wne = deepcopy(cfg_uniform_25_swinir_o)
cfg_uniform_25_swinir_wne.lr_halving_steps = int(1e5)
cfg_uniform_25_swinir_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
_set_s_samples_per_epoch(cfg_uniform_25_swinir_wne)

# Rayleigh @17 dB
cfg_rayleigh_17_swinir_o = deepcopy(cfg_rayleigh_17)
cfg_rayleigh_17_swinir_o.model = "swinir"
cfg_rayleigh_17_swinir_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_rayleigh_17_swinir_o.s_patch_size = swinir_patch_size
cfg_rayleigh_17_swinir_o.batch_size = swinir_batch_size
_set_s_samples_per_epoch(cfg_rayleigh_17_swinir_o)

cfg_rayleigh_17_swinir_wne = deepcopy(cfg_rayleigh_17_swinir_o)
cfg_rayleigh_17_swinir_wne.lr_halving_steps = int(1e5)
cfg_rayleigh_17_swinir_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
_set_s_samples_per_epoch(cfg_rayleigh_17_swinir_wne)

# %% Restormer experiments

# Restormer configs: ordinary (_o) and WNE (_wne)
restormer_patch_size = 64
restormer_batch_size = 8

cfg_50_restormer_o = deepcopy(cfg_50)
cfg_50_restormer_o.model = "restormer"
cfg_50_restormer_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_50_restormer_o.s_patch_size = restormer_patch_size
cfg_50_restormer_o.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_50_restormer_o)

cfg_25_restormer_o = deepcopy(cfg_25)
cfg_25_restormer_o.model = "restormer"
cfg_25_restormer_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_25_restormer_o.s_patch_size = restormer_patch_size
cfg_25_restormer_o.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_25_restormer_o)

cfg_10_restormer_o = deepcopy(cfg_10)
cfg_10_restormer_o.model = "restormer"
cfg_10_restormer_o.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_10_restormer_o.s_patch_size = restormer_patch_size
cfg_10_restormer_o.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_10_restormer_o)

cfg_50_restormer_wne = deepcopy(cfg_50)
cfg_50_restormer_wne.model = "restormer"
cfg_50_restormer_wne.lr_halving_steps = int(1e5)
cfg_50_restormer_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_50_restormer_wne.s_patch_size = restormer_patch_size
cfg_50_restormer_wne.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_50_restormer_wne)

cfg_25_restormer_wne = deepcopy(cfg_25)
cfg_25_restormer_wne.model = "restormer"
cfg_25_restormer_wne.lr_halving_steps = int(1e5)
cfg_25_restormer_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_25_restormer_wne.s_patch_size = restormer_patch_size
cfg_25_restormer_wne.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_25_restormer_wne)

cfg_10_restormer_wne = deepcopy(cfg_10)
cfg_10_restormer_wne.model = "restormer"
cfg_10_restormer_wne.lr_halving_steps = int(1e5)
cfg_10_restormer_wne.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_10_restormer_wne.s_patch_size = restormer_patch_size
cfg_10_restormer_wne.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_10_restormer_wne)

# Single-vs-multi noise study for the paper framing.
# Keep the current single-noise control and broaden only to the existing
# experimental span U[0, 55].
study_multi_noise_range_8bit = (0.0, 55.0)
cfg_study_swinir_baseline_single = deepcopy(cfg_25_swinir_o)
cfg_study_swinir_wne_single = deepcopy(cfg_25_swinir_wne)

cfg_study_swinir_baseline_multi = _make_backbone_cfg(
    _with_noise_range(deepcopy(cfg), *study_multi_noise_range_8bit),
    model="swinir",
    wrapper_mode="idem",
    patch_size=swinir_patch_size,
    batch_size=swinir_batch_size,
)

cfg_study_swinir_wne_multi = _make_backbone_cfg(
    _with_noise_range(deepcopy(cfg), *study_multi_noise_range_8bit),
    model="swinir",
    wrapper_mode="norm-equiv",
    patch_size=swinir_patch_size,
    batch_size=swinir_batch_size,
    lr_halving_steps=int(1e5),
)

# Restormer RGB configs: ordinary (_o_rgb) and WNE (_wne_rgb)
cfg_50_color = _with_color_data(deepcopy(cfg_50))
cfg_25_color = _with_color_data(deepcopy(cfg_25))
cfg_10_color = _with_color_data(deepcopy(cfg_10))

cfg_50_restormer_o_rgb = deepcopy(cfg_50_color)
cfg_50_restormer_o_rgb.model = "restormer"
cfg_50_restormer_o_rgb.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_50_restormer_o_rgb.s_patch_size = restormer_patch_size
cfg_50_restormer_o_rgb.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_50_restormer_o_rgb)

cfg_50_restormer_wne_rgb = deepcopy(cfg_50_color)
cfg_50_restormer_wne_rgb.model = "restormer"
cfg_50_restormer_wne_rgb.lr_halving_steps = int(1e5)
cfg_50_restormer_wne_rgb.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_50_restormer_wne_rgb.s_patch_size = restormer_patch_size
cfg_50_restormer_wne_rgb.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_50_restormer_wne_rgb)

cfg_25_restormer_o_rgb = deepcopy(cfg_25_color)
cfg_25_restormer_o_rgb.model = "restormer"
cfg_25_restormer_o_rgb.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_25_restormer_o_rgb.s_patch_size = restormer_patch_size
cfg_25_restormer_o_rgb.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_25_restormer_o_rgb)

cfg_25_restormer_wne_rgb = deepcopy(cfg_25_color)
cfg_25_restormer_wne_rgb.model = "restormer"
cfg_25_restormer_wne_rgb.lr_halving_steps = int(1e5)
cfg_25_restormer_wne_rgb.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_25_restormer_wne_rgb.s_patch_size = restormer_patch_size
cfg_25_restormer_wne_rgb.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_25_restormer_wne_rgb)

cfg_10_restormer_o_rgb = deepcopy(cfg_10_color)
cfg_10_restormer_o_rgb.model = "restormer"
cfg_10_restormer_o_rgb.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="idem"
)
cfg_10_restormer_o_rgb.s_patch_size = restormer_patch_size
cfg_10_restormer_o_rgb.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_10_restormer_o_rgb)

cfg_10_restormer_wne_rgb = deepcopy(cfg_10_color)
cfg_10_restormer_wne_rgb.model = "restormer"
cfg_10_restormer_wne_rgb.lr_halving_steps = int(1e5)
cfg_10_restormer_wne_rgb.model_cfg = ModelConfig(
    model_mode="ordinary", pred_mode="direct", wrapper_mode="norm-equiv"
)
cfg_10_restormer_wne_rgb.s_patch_size = restormer_patch_size
cfg_10_restormer_wne_rgb.batch_size = restormer_batch_size
_set_s_samples_per_epoch(cfg_10_restormer_wne_rgb)
