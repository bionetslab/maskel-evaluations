# maskel-evaluations

Benchmarks, real-data regression tests, and analysis notebooks for [maskel](https://github.com/bionetslab/maskel). Not a published package — this repo just consumes `maskel` (and, for the napari-specific notebooks/scripts, [napari-maskel](https://github.com/bionetslab/napari-maskel)) from PyPI.

## Installation

```sh
uv sync --extra dev --extra notebooks
```

Before `maskel` has its first PyPI release, point uv at a local checkout instead: `uv sync && uv pip install -e ../maskel` (adjust the path), or add a local, untracked `uv.toml` with a `[tool.uv.sources]` override for `maskel`.

## Dataset

Benchmarks and regression tests here run against the High-Resolution Fundus (HRF) Image Database — 45 retinal fundus images (15 healthy, 15 diabetic retinopathy, 15 glaucoma) with binary gold-standard vessel segmentations and field-of-view masks. It's ~107MB and not checked into this repo (see `.gitignore`); download it from the [HRF Image Database](https://www5.cs.fau.de/research/data/fundus-images/) and place it at `HRF/` (containing `images/`, `manual1/`, `mask/`) before running anything that depends on it.

> Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.
> Robust Vessel Segmentation in Fundus Images.
> International Journal of Biomedical Imaging, vol. 2013, 2013

The HRF dataset is released under the **Creative Commons 4.0 Attribution License**, separate from this repo's MIT license (see [LICENSE](LICENSE)).

## Benchmarks

```sh
python benchmark/2d_comparison.py        # maskel vs skimage vs VesselVio, 2D (HRF)
python benchmark/3d_comparison.py        # maskel vs skimage vs VesselVio, 3D (VESSAP volume)
python benchmark/3d_lung_comparison.py   # maskel vs skimage vs VesselVio, 3D (VESSEL12 lung scans, auto-downloaded)
```

The VesselVio comparisons need a local [VesselVio](https://github.com/JacobBumgarner/VesselVio) checkout — set `VESSELVIO_PATH` to it before running. `3d_comparison.py` reads `data/vessap_preprocessed.tif` by default (override with `VESSAP_VOLUME_PATH`).

## Tests

```sh
uv sync --extra dev && pytest                # all tests (needs HRF/ present)
uv sync --extra dev && pytest -m "not slow"  # skip the HRF-based regression tests
```

- **2D regression** - thinning + feature extraction on all 45 HRF samples, compared against saved baselines in `tests/skeletons/` and `tests/features/`
- **2D comparison** - maskel `lee94_thin` vs `skimage.morphology.skeletonize` on one HRF sample, asserting identical output

First run (or `--update-baseline`) generates baselines. The equivalent self-contained (no external dataset) 3D regression tests live in the `maskel` repo itself.

## Analysis notebooks

`analysis/HRF_Feature_Analysis.ipynb` and `analysis/HRF_Prediction.ipynb` run the full feature-extraction pipeline over the HRF dataset and explore phenotype (healthy / diabetic retinopathy / glaucoma) differences and classification. They need `HRF/` present and the `notebooks` extra installed.

## License

This repo's code is released under the **MIT License**. See [LICENSE](LICENSE) for details — note the HRF dataset itself carries a separate CC-BY 4.0 license (see Dataset section above).
