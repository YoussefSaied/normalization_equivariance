from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable


def find_train_psnr_files(root: Path) -> Iterable[Path]:
    """Yield all *_train_psnr.csv files under the eval_logs tree."""
    yield from root.glob("eval_logs/*/*_train_psnr.csv")


def parse_filename(path: Path) -> tuple[str, str]:
    """
    Return (arch, sigma_str) from filenames like dncnn_sigma10_train_psnr.csv.

    Raises ValueError if the filename does not match the expected pattern.
    """
    m = re.match(r"(?P<arch>[^_]+)_sigma(?P<sigma>\d+)_train_psnr\.csv$", path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path}")
    return m.group("arch"), m.group("sigma")


def combine(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in find_train_psnr_files(root):
        dataset = csv_path.parent.name
        arch, sigma = parse_filename(csv_path)

        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "dataset": dataset,
                        "arch": arch,
                        "sigma_8bit": sigma,
                        "model": row["model"],
                        "train_sigma_8bit": row["train_sigma_8bit"],
                        "train_sigma": row["train_sigma"],
                        "psnr_db": row["psnr_db"],
                    }
                )

    rows.sort(
        key=lambda r: (
            r["dataset"],
            r["arch"],
            float(r["train_sigma"]),
            r["model"],
        )
    )
    return rows


def write_combined(rows: list[dict[str, str]], dest: Path) -> None:
    fieldnames = [
        "dataset",
        "arch",
        "sigma_8bit",
        "model",
        "train_sigma_8bit",
        "train_sigma",
        "psnr_db",
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    rows = combine(root)
    output_path = root / "eval_logs" / "combined_train_psnr.csv"
    write_combined(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
