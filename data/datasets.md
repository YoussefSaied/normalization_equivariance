# Datasets

Included for quick evaluation:

- `Set12/`: test images used by most mismatch figures.
- `Set68/`: BSD68 test images.

Not bundled:

- `BSD400/`: training images used by the fixed-noise denoising configs.
- `CBSD68/`: color test images for color Restormer evaluation.
- `WaterlooExploration/`
- `DIV2K/`
- `Flickr2K/`
- `SIDD/`

`python data/download_datasets.py` can fetch Waterloo, DIV2K, Flickr2K, and SIDD. Place BSD400 and CBSD68 manually under the directory names above.

Training configs in `experiments_cfg.py` assume these directory names.
