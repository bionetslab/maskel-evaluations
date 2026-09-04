# maskel-evaluations
Simon Wittmann, Anna Möller

Benchmarks and analysis notebooks for [maskel](https://github.com/bionetslab/maskel/), the vessel skeletonization and phenotype feature extraction package. Not a published package itself — this repo just consumes `maskel` from PyPI. See also [napari-maskel](https://github.com/bionetslab/napari-maskel/), the interactive napari plugin built on the same package.

## Installation

```sh
uv sync --extra notebooks
```

## Datasets

Benchmarks here run against two public vessel-segmentation datasets, neither of which is checked into this repo (see `.gitignore`). Both are auto-downloaded into `data/` on first use — nothing to set up manually to just run the benchmarks.

**HRF** — the High-Resolution Fundus (HRF) Image Database: 45 retinal fundus images (15 healthy, 15 diabetic retinopathy, 15 glaucoma) with binary gold-standard vessel segmentations. `benchmark/hrf.py`'s `ensure_hrf()` downloads the three manual-segmentation archives from the [HRF Image Database](https://www5.cs.fau.de/research/data/fundus-images/) into `data/HRF/manual1/` — that's the only part of the dataset the benchmarks use. The raw fundus images and field-of-view masks aren't fetched automatically; if you want the full dataset (e.g. for the notebooks in `tutorials/`, which use the FOV mask), download it yourself from the link above and place `images/` and `mask/` alongside `manual1/` under `data/HRF/`.

> Budai, Attila; Bock, Rüdiger; Maier, Andreas; Hornegger, Joachim; Michelson, Georg.
> Robust Vessel Segmentation in Fundus Images.
> International Journal of Biomedical Imaging, vol. 2013, 2013

The HRF dataset is released under the **Creative Commons 4.0 Attribution License**, separate from this repo's MIT license (see [LICENSE](LICENSE)).

**VESSEL12** — 20 thoracic CT lung-mask scans. `benchmark/vessel12.py`'s `ensure_vessel12()` downloads them from [Zenodo](https://zenodo.org/records/8055066) into `data/Vessel12/` on first use.

## Benchmarks

```sh
uv run python benchmark/2d_HRF_comparison.py   # maskel vs skimage vs VesselVio, timing, HRF
uv run python benchmark/2d_HRF_memory.py       # maskel vs skimage vs VesselVio, peak RAM, HRF
uv run python benchmark/3d_lung_comparison.py  # maskel vs skimage vs VesselVio, timing, VESSEL12
uv run python benchmark/3d_lung_memory.py      # maskel vs skimage vs VesselVio, peak RAM, VESSEL12
```

(`3d_lung_*` is named for the VESSEL12 lung CT scans it benchmarks — see Datasets above.)

The VesselVio comparisons need a local [VesselVio](https://github.com/JacobBumgarner/VesselVio) checkout — set `VESSELVIO_PATH` to it before running.

Every benchmark writes its results to a CSV in `results/`: `2d_HRF_runtime.csv`, `2d_HRF_memory.csv`, `3d_lung_runtime.csv`, `3d_lung_memory.csv`. `benchmark/paper_table.py` reads all four and writes `results/runtime_table.tex` and `results/memory_table.tex`. These committed CSVs and `.tex` tables in `results/` are the actual numbers reported in the paper, not just example output.

The HRF benchmarks finish in minutes; the VESSEL12 ones do not — the timing comparison (20 scans x 5 repeats x 3 methods) took ~2.5h on a 64-core (128-thread) exclusive node. `slurm/*.sbatch` are the job scripts used to actually produce the committed `results/`; submit them on an HPC cluster (`sbatch slurm/run_3d_lung_comparison_benchmark.sbatch`, etc.) rather than running the VESSEL12 scripts interactively.

## Analysis notebooks

`tutorials/HRF_Feature_Analysis.ipynb` and `tutorials/HRF_Prediction.ipynb` run the full feature-extraction pipeline over the HRF dataset and explore phenotype (healthy / diabetic retinopathy / glaucoma) differences and classification. They need the `notebooks` extra installed and the full HRF dataset present at `data/HRF/` (including `mask/` — see Datasets above).

## Citation

If you use `maskel` or these benchmarks, please cite:

> TODO: preprint citation and DOI

## License

This repo's code is released under the **MIT License**. See [LICENSE](LICENSE) for details — note the HRF dataset itself carries a separate CC-BY 4.0 license (see Datasets section above).
