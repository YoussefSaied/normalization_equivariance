from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from gdown.download import download as gdownload
import wget


DATA_DIR = Path(__file__).resolve().parent

WATERLOO_LINK = (
    "https://ivc.uwaterloo.ca/database/WaterlooExploration/waterloo_exploration.rar"
)
DIV2K_TRAIN_LINK = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_LINK = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
FLICKR2K_LINK = "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"

# Official SIDD dataset homepage for provenance and dataset structure:
# https://abdokamel.github.io/sidd/
SIDD_INDEX_URL = "https://abdokamel.github.io/sidd/"

# SIDD paper: Abdelhamed, Lin, and Brown, "A High-Quality Denoising Dataset for Smartphone Cameras," CVPR 2018.

# SIDD training archive. We download the public `train.zip` archive and normalize its
# extracted scene folders into `data/SIDD/Data/`.
# Citation: MMagic "Preparing SIDD Dataset" points the SIDD training set to this file:
# https://mmagic.readthedocs.io/en/stable/dataset_zoo/sidd.html
# Direct file page:
# https://drive.google.com/file/d/1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw/view?usp=sharing
SIDD_TRAIN_DOWNLOAD_URL = (
    "https://drive.google.com/uc?id=1UHjWZzLPGweA9ZczmV8lFSRcIxqiOVJw"
)

# SIDD benchmark MAT archive. We download the official SIDD evaluation blocks and store
# `ValidationNoisyBlocksSrgb.mat` and `ValidationGtBlocksSrgb.mat` under `data/SIDD/`.
# Citation: Restormer's public SIDD downloader uses this file as `SIDD_test`, and
# MMagic documents the benchmark artifact as these MAT files:
# https://gitextract.com/swz30/Restormer
# https://mmagic.readthedocs.io/en/stable/dataset_zoo/sidd.html
# Direct file page:
# https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view?usp=sharing
SIDD_TEST_MAT_DOWNLOAD_URL = (
    "https://drive.google.com/uc?id=11vfqV-lqousZTuAit1Qkqghiv_taY0KZ"
)
SIDD_TEST_MAT_FILENAMES = (
    "ValidationNoisyBlocksSrgb.mat",
    "ValidationGtBlocksSrgb.mat",
)


def _gb_progress_bar(current, total, width=40):
    """Display wget progress in GB."""
    if total <= 0:
        total = current or 1
    ratio = min(current / total, 1)
    filled = int(width * ratio)
    bar = "=" * filled + " " * (width - filled)
    downloaded = current / (1024**3)
    total_gb = total / (1024**3)
    print(
        f"\r[{bar}] {downloaded:.2f}/{total_gb:.2f} GB",
        end="",
        flush=True,
    )
    if current >= total:
        print()


def _dataset_dir(name: str) -> Path:
    return DATA_DIR / name


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Found existing file, skipping download: {destination.name}")
        return destination
    print(f"Downloading {destination.name}...")
    wget.download(url, str(destination), bar=_gb_progress_bar)
    print()
    return destination


def _download_google_drive_file(download_url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Found existing file, skipping download: {destination.name}")
        return destination

    print(f"Downloading {destination.name} from Google Drive via gdown...")
    downloaded_path = gdownload(
        url=download_url,
        output=str(destination),
        quiet=False,
    )
    if downloaded_path is None or not destination.exists():
        raise RuntimeError(f"gdown failed to download {destination.name}.")
    return destination


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(destination)
        return
    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(path=destination)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def _extract_waterloo_with_unrar(rar_path: Path, destination: Path) -> None:
    if shutil.which("unrar") is None:
        raise RuntimeError(
            "The 'unrar' command is required for Waterloo extraction but was not found."
        )
    cmd = ["unrar", "x", "-idq", "-o+", str(rar_path)]
    subprocess.run(cmd, cwd=destination, check=True)


def _reset_extract_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _contains_sidd_scene_instances(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    child_dirs = [child for child in directory.iterdir() if child.is_dir()]
    if not child_dirs:
        return False
    return any(
        any(child.glob("*GT_SRGB*.PNG")) or any(child.glob("*GT_SRGB*.png"))
        for child in child_dirs
    )


def _resolve_sidd_data_dir(extract_dir: Path) -> Path | None:
    preferred = extract_dir / "train"
    if _contains_sidd_scene_instances(preferred):
        return preferred
    if _contains_sidd_scene_instances(extract_dir):
        return extract_dir
    for candidate in extract_dir.rglob("*"):
        if _contains_sidd_scene_instances(candidate):
            return candidate
    return None


def _has_sidd_test_mats(dataset_root: Path) -> bool:
    return all(
        (dataset_root / filename).is_file() for filename in SIDD_TEST_MAT_FILENAMES
    )


def _download_and_extract_sidd_train(dataset_root: Path) -> None:
    tmp_dir = dataset_root.with_name(f"{dataset_root.name}_train_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    archive_path = _download_google_drive_file(
        SIDD_TRAIN_DOWNLOAD_URL,
        tmp_dir / "train.zip",
    )
    extract_dir = tmp_dir / "extract"
    _reset_extract_dir(extract_dir)
    _extract_archive(archive_path, extract_dir)

    data_dir = _resolve_sidd_data_dir(extract_dir)
    if data_dir is None:
        raise FileNotFoundError(
            "Downloaded SIDD training archive did not contain the expected scene folders."
        )

    target_data_dir = dataset_root / "Data"
    if target_data_dir.exists():
        shutil.rmtree(target_data_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(data_dir), str(target_data_dir))
    shutil.rmtree(tmp_dir)


def _download_and_extract_sidd_test_mats(dataset_root: Path) -> None:
    tmp_dir = dataset_root.with_name(f"{dataset_root.name}_test_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    archive_path = _download_google_drive_file(
        SIDD_TEST_MAT_DOWNLOAD_URL,
        tmp_dir / "sidd_test.zip",
    )
    extract_dir = tmp_dir / "extract"
    _reset_extract_dir(extract_dir)
    _extract_archive(archive_path, extract_dir)

    located_files = {
        filename: next(extract_dir.rglob(filename), None)
        for filename in SIDD_TEST_MAT_FILENAMES
    }
    missing = [name for name, path in located_files.items() if path is None]
    if missing:
        raise FileNotFoundError(
            "Downloaded SIDD benchmark archive did not contain: " + ", ".join(missing)
        )

    dataset_root.mkdir(parents=True, exist_ok=True)
    for filename, source in located_files.items():
        destination = dataset_root / filename
        if destination.exists():
            destination.unlink()
        assert source is not None
        shutil.move(str(source), str(destination))
    shutil.rmtree(tmp_dir)


def waterloo_exploration() -> None:
    dataset_dir = _dataset_dir("WaterlooExploration")
    if dataset_dir.is_dir() and any(dataset_dir.iterdir()):
        print("Waterloo Exploration Dataset already exists.")
        return

    tmp_dir = dataset_dir.with_name(f"{dataset_dir.name}_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rar_path = DATA_DIR / "waterloo_exploration.rar"
    if rar_path.exists():
        print("Found existing Waterloo archive, skipping download.")
    else:
        _download_file(WATERLOO_LINK, rar_path)

    print("Extracting files with unrar...")
    start_time = time.perf_counter()
    _extract_waterloo_with_unrar(rar_path, tmp_dir)
    elapsed_time = time.perf_counter() - start_time
    print(f"Extraction completed in {elapsed_time:.2f} seconds.")

    pristine_dir = tmp_dir / "pristine_images"
    if pristine_dir.is_dir():
        for entry in pristine_dir.iterdir():
            shutil.move(str(entry), str(tmp_dir / entry.name))
        shutil.rmtree(pristine_dir)

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.replace(dataset_dir)

    try:
        rar_path.unlink()
    except OSError:
        pass
    print("Download and extraction complete.")


def div2k() -> None:
    combined_dir = _dataset_dir("DIV2K")

    if combined_dir.is_dir() and any(combined_dir.iterdir()):
        print("DIV2K dataset already combined.")
        return

    splits = (
        ("training", DIV2K_TRAIN_LINK, _dataset_dir("DIV2K_train_HR")),
        ("validation", DIV2K_VALID_LINK, _dataset_dir("DIV2K_valid_HR")),
    )

    for name, url, split_dir in splits:
        if split_dir.is_dir() and any(split_dir.iterdir()):
            print(f"DIV2K {name} split already available.")
            continue

        zip_path = DATA_DIR / Path(urlparse(url).path).name
        _download_file(url, zip_path)
        print(f"Extracting DIV2K {name} split...")
        _extract_archive(zip_path, DATA_DIR)
        zip_path.unlink(missing_ok=True)
        print(f"{name.title()} split ready.")

    combined_dir.mkdir(parents=True, exist_ok=True)
    for _, _, split_dir in splits:
        if not split_dir.is_dir():
            continue
        for entry in split_dir.iterdir():
            shutil.move(str(entry), str(combined_dir / entry.name))
        shutil.rmtree(split_dir)

    print("Combined DIV2K training and validation splits into DIV2K/.")


def flickr2k() -> None:
    dataset_dir = _dataset_dir("Flickr2K")
    if dataset_dir.is_dir() and any(dataset_dir.iterdir()):
        print("Flickr2K Dataset already exists.")
        return

    tar_path = DATA_DIR / "Flickr2K.tar"
    if not tar_path.exists():
        print("Downloading Flickr2K Dataset:")
        print(
            "nohup aria2c -c -x16 -s16 "
            f"{FLICKR2K_LINK} -o Flickr2K.tar > flickr2k.log 2>&1 &"
        )
        print("tail -f flickr2k.log")
        return

    print("Extracting Flickr2K...")
    _extract_archive(tar_path, DATA_DIR)
    tar_path.unlink(missing_ok=True)

    for entry in dataset_dir.iterdir():
        if entry.name.endswith("_LR"):
            shutil.rmtree(entry)
    print("Flickr2K Dataset download and extraction complete.")


def sidd() -> None:
    dataset_root = _dataset_dir("SIDD")
    train_dir = dataset_root / "Data"
    test_ready = _has_sidd_test_mats(dataset_root)

    if train_dir.is_dir() and test_ready:
        print("SIDD training data and benchmark MAT files already exist.")
        return

    dataset_root.mkdir(parents=True, exist_ok=True)

    if train_dir.is_dir():
        print("SIDD training data already exists.")
    else:
        _download_and_extract_sidd_train(dataset_root)

    if not test_ready:
        _download_and_extract_sidd_test_mats(dataset_root)

    print(
        "SIDD download complete. Training data is under SIDD/Data and benchmark "
        "MAT files are stored under SIDD/."
    )


DATASETS: dict[str, Callable[[], None]] = {
    "div2k": div2k,
    "flickr2k": flickr2k,
    "sidd": sidd,
    "waterloo_exploration": waterloo_exploration,
}


def main(selected: list[str] | None = None) -> None:
    dataset_names = selected or ["all"]
    if "all" in dataset_names:
        dataset_names = list(DATASETS)
    for name in dataset_names:
        DATASETS[name]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets into data/.")
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=["all", *sorted(DATASETS)],
        default=["all"],
        help="Datasets to download. Defaults to all registered datasets.",
    )
    args = parser.parse_args()
    main(args.datasets)
