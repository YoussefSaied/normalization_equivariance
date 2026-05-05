# reference implementation: https://github.com/SaoYan/DnCNN-PyTorch/blob/master/dataset.py
import os
import os.path
from dataclasses import dataclass, field

import numpy as np
import h5py
import cv2
import glob
import tqdm


@dataclass
class DataConfig:
    data_path: str = "."
    patch_size: int = 50
    stride: int = 10
    aug_times: int = 2
    scales: list[float] = field(default_factory=lambda: [1, 0.9, 0.8, 0.7])
    train_dirs: list[str] = field(default_factory=lambda: ["BSD400"])
    valid_dirs: list[str] = field(default_factory=lambda: ["Set12"])


def Im2Patch(img: np.ndarray, win: int, stride: int = 1) -> np.ndarray:
    k = 0
    c = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0 : endw - win + 0 + 1 : stride, 0 : endh - win + 0 + 1 : stride]
    total_num_patches = patch.shape[1] * patch.shape[2]
    Y = np.zeros([c, win * win, total_num_patches], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[
                :, i : endw - win + i + 1 : stride, j : endh - win + j + 1 : stride
            ]
            Y[:, k, :] = np.array(patch[:]).reshape(c, total_num_patches)
            k = k + 1
    return Y.reshape([c, win, win, total_num_patches])


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def collect_image_files(base_path: str, folders: list[str]) -> list[str]:
    """Return sorted list of PNG files from each folder (relative or absolute)."""
    files: list[str] = []
    for folder in folders:
        folder_path = (
            folder if os.path.isabs(folder) else os.path.join(base_path, folder)
        )
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")
        files.extend(sorted(glob.glob(os.path.join(folder_path, "*.png"))))
    return files


def generate_image_patches(
    cfg: DataConfig, h5f: h5py.File, train_size: int, file: str
) -> int:
    img = cv2.imread(file)
    scales = cfg.scales
    h, w, c = img.shape
    for k in range(len(scales)):
        Img = cv2.resize(
            img,
            (int(h * scales[k]), int(w * scales[k])),
            interpolation=cv2.INTER_CUBIC,
        )
        Img = np.expand_dims(Img[:, :, 0].copy(), 0) / 255.0
        patches = Im2Patch(Img, win=cfg.patch_size, stride=cfg.stride)
        for n in range(patches.shape[3]):
            data = patches[:, :, :, n].copy()
            h5f.create_dataset(str(train_size), data=data)
            train_size += 1
            for m in range(cfg.aug_times - 1):
                data_aug = data_augmentation(data, np.random.randint(1, 8))
                h5f.create_dataset(str(train_size) + "_aug_%d" % (m + 1), data=data_aug)
                train_size += 1
    return train_size


def main(cfg: DataConfig):
    print("Processing training data")

    files = collect_image_files(cfg.data_path, cfg.train_dirs)
    with h5py.File(os.path.join(cfg.data_path, "train.h5"), "w") as h5f:
        train_size = 0
        for file in tqdm.tqdm(files):
            train_size = generate_image_patches(cfg, h5f, train_size, file)

    print("Processing validation data")
    files = collect_image_files(cfg.data_path, cfg.valid_dirs)
    with h5py.File(os.path.join(cfg.data_path, "valid.h5"), "w") as h5f:
        valid_size = 0
        for file in tqdm.tqdm(files):
            img = cv2.imread(file)
            img = np.expand_dims(img[:, :, 0], 0) / 255.0
            h5f.create_dataset(str(valid_size), data=img)
            valid_size += 1

    print(f"Training size {train_size}, validation size {valid_size}")


if __name__ == "__main__":
    cfg = DataConfig()
    main(cfg)
