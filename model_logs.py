from pathlib import Path

# Release-local checkpoint registry. Keep these paths relative so the repository
# can be cloned anywhere without depending on the author's workstation layout.
models_log = {
    "ne_fdncnn_50": Path("logs/ne_fdncnn_50"),
    "ne_fdncnn_25": Path("logs/ne_fdncnn_25"),
    "ne_fdncnn_10": Path("logs/ne_fdncnn_10"),

    "se_fdncnn_50": Path("logs/se_fdncnn_50"),
    "se_fdncnn_25": Path("logs/se_fdncnn_25"),
    "se_fdncnn_10": Path("logs/se_fdncnn_10"),

    "wne_dncnn_50": Path("logs/wne_dncnn_50"),
    "wne_dncnn_25": Path("logs/wne_dncnn_25"),
    "wne_dncnn_10": Path("logs/wne_dncnn_10"),
    "wnei_dncnn_10": Path("logs/wnei_dncnn_10"),

    "b_dncnn_50": Path("logs/b_dncnn_50"),
    "b_dncnn_25": Path("logs/b_dncnn_25"),
    "b_dncnn_10": Path("logs/b_dncnn_10"),

    "b_swinir_50": Path("logs/b_swinir_50"),
    "b_swinir_25": Path("logs/b_swinir_25"),
    "b_swinir_10": Path("logs/b_swinir_10"),

    "wne_swinir_50": Path("logs/wne_swinir_50"),
    "wne_swinir_25": Path("logs/wne_swinir_25"),
    "wne_swinir_10": Path("logs/wne_swinir_10"),

    "b_swinir_0-55": Path("logs/b_swinir_0-55"),
    "wne_swinir_0-55": Path("logs/wne_swinir_0-55"),
    "softne_swinir_10": Path("logs/softne_swinir_10"),
    "b_swinir_10_n2n": Path("logs/b_swinir_10_n2n"),
    "wne_swinir_10_n2n": Path("logs/wne_swinir_10_n2n"),

    "rayleigh_wne_dncnn_l1_17": Path("logs/rayleigh_wne_dncnn_l1_17"),
    "rayleigh_ne_fdncnn_l1_17": Path("logs/rayleigh_ne_fdncnn_l1_17"),
    "rayleigh_se_fdncnn_l1_17": Path("logs/rayleigh_se_fdncnn_l1_17"),
    "rayleigh_b_dncnn_l1_17": Path("logs/rayleigh_b_dncnn_l1_17"),

    "laplace_wne_dncnn_l1_20": Path("logs/laplace_wne_dncnn_l1_20"),
    "laplace_ne_fdncnn_l1_20": Path("logs/laplace_ne_fdncnn_l1_20"),
    "laplace_se_fdncnn_l1_20": Path("logs/laplace_se_fdncnn_l1_20"),
    "laplace_b_dncnn_l1_20": Path("logs/laplace_b_dncnn_l1_20"),

    "uniform_wne_dncnn_l1_14": Path("logs/uniform_wne_dncnn_l1_14"),
    "uniform_ne_fdncnn_l1_14": Path("logs/uniform_ne_fdncnn_l1_14"),
    "uniform_se_fdncnn_l1_14": Path("logs/uniform_se_fdncnn_l1_14"),
    "uniform_b_dncnn_l1_14": Path("logs/uniform_b_dncnn_l1_14"),
}
