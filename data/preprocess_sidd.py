from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from PIL import Image
from tqdm import tqdm

try:
    from data.sidd_dataset import (
        SIDD_MEDIUM_DATA_DIR,
        SIDD_PATCHES_DIR,
        find_sidd_medium_srgb_pairs,
        read_rgb_image,
    )
except ModuleNotFoundError:  # Supports `python data/preprocess_sidd.py`.
    from sidd_dataset import (  # type: ignore[no-redef]
        SIDD_MEDIUM_DATA_DIR,
        SIDD_PATCHES_DIR,
        find_sidd_medium_srgb_pairs,
        read_rgb_image,
    )


@dataclass
class SIDDPreprocessConfig:
    src_dir: str = str(SIDD_MEDIUM_DATA_DIR)
    dst_dir: str = str(SIDD_PATCHES_DIR)
    patch_size: int = 512
    stride: int = 384
    overwrite: bool = False


def sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        raise ValueError(
            f"Patch size {patch_size} exceeds image extent {length}. "
            "SIDD full-resolution images should be larger than 512."
        )

    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def save_patch(image, path: Path) -> None:
    Image.fromarray(image).save(path, compress_level=1)


def preprocess_sidd_patches(
    config: SIDDPreprocessConfig | None = None,
) -> dict[str, str | int]:
    cfg = config or SIDDPreprocessConfig()
    src_dir = Path(cfg.src_dir)
    dst_dir = Path(cfg.dst_dir)
    pairs = find_sidd_medium_srgb_pairs(src_dir)

    if cfg.overwrite and dst_dir.exists():
        shutil.rmtree(dst_dir)

    input_dir = dst_dir / "input"
    gt_dir = dst_dir / "gt"
    input_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    patch_count = 0
    for noisy_path, gt_path in tqdm(pairs, desc="Tiling SIDD", unit="pair"):
        noisy = read_rgb_image(noisy_path)
        gt = read_rgb_image(gt_path)
        if noisy.shape != gt.shape:
            raise ValueError(
                f"Mismatched pair shapes for {noisy_path.name}: {noisy.shape} vs {gt.shape}"
            )

        height, width = noisy.shape[:2]
        ys = sliding_positions(height, cfg.patch_size, cfg.stride)
        xs = sliding_positions(width, cfg.patch_size, cfg.stride)

        stem_prefix = f"{gt_path.parent.name}__{gt_path.stem}"
        for top, left in product(ys, xs):
            patch_name = f"{stem_prefix}__y{top:04d}_x{left:04d}.png"
            noisy_patch = noisy[top : top + cfg.patch_size, left : left + cfg.patch_size]
            gt_patch = gt[top : top + cfg.patch_size, left : left + cfg.patch_size]
            save_patch(noisy_patch, input_dir / patch_name)
            save_patch(gt_patch, gt_dir / patch_name)
            patch_count += 1

    metadata: dict[str, str | int] = {
        "src_dir": str(src_dir.resolve()),
        "num_pairs": len(pairs),
        "num_patches": patch_count,
        "patch_size": cfg.patch_size,
        "stride": cfg.stride,
        "format": "png",
        "input_dir": "input",
        "gt_dir": "gt",
    }
    with (dst_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main(config: SIDDPreprocessConfig | None = None) -> None:
    metadata = preprocess_sidd_patches(config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
