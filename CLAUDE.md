# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

maskel-evaluations: benchmarks, real-data regression tests, and analysis notebooks for [maskel](https://github.com/bionetslab/maskel). Not a published package — `[tool.uv] package = false` in `pyproject.toml`. Code here consumes `maskel` as an ordinary dependency; nothing in `maskel` depends on this repo.

## Commands

```sh
uv sync --extra dev --extra notebooks
uv run pytest                    # needs HRF/ present, see README
uv run pytest -m "not slow"      # skip HRF-based regression tests
uv run pytest --update-baseline  # regenerate tests/skeletons/, tests/features/
python benchmark/2d_comparison.py
```

## Architecture

**Local `hrf.py`, not a `maskel` module.** `HRFDataset`/`preprocess_segmentation` live at the repo root, not inside `maskel` — they're dataset-loading glue specific to this one benchmark dataset, not general skeletonization API. Scripts/tests import it as `from hrf import HRFDataset, preprocess_segmentation`, which works because `pyproject.toml` sets `[tool.pytest.ini_options] pythonpath = ["."]` for tests, and `benchmark/*.py` insert the repo root onto `sys.path` themselves before importing it (they're run directly with `python benchmark/foo.py`, not through pytest).

**Benchmark scripts need external inputs, deliberately not hardcoded.** `VESSELVIO_PATH` (env var) must point at a local [VesselVio](https://github.com/JacobBumgarner/VesselVio) checkout for the three `benchmark/*_comparison.py` scripts — they raise a clear `SystemExit` if it's unset rather than silently failing on a machine-specific path. `benchmark/3d_comparison.py`'s `VESSAP_VOLUME_PATH` defaults to `data/vessap_preprocessed.tif` in this repo but is also overridable. `benchmark/3d_lung_comparison.py` auto-downloads its VESSEL12 lung masks from Zenodo into `tests/lung_masks/` (gitignored) on first run — no manual setup needed there.

**The HRF dataset itself is gitignored.** `HRF/` is ~107MB of JPEG/TIFF images and is *not* committed (see `.gitignore`) — download it separately per the README before running anything that needs it. The regression-test baselines derived from it (`tests/skeletons/skeleton_*.npz`, `tests/features/features_*.csv`) are small and *are* committed, since they're the actual regression ground truth.

**Mirrors `maskel`'s regression-test pattern, for real data.** `tests/test_2d_thinning_regression.py` and `tests/test_2d_skimage_comparison.py` are the HRF counterparts of the self-contained (brain-volume) regression tests that stayed in the `maskel` repo — same baseline/`--update-baseline` convention, via a local `tests/_helpers.py` (a trimmed copy, not shared across repos).

## Memory benchmarks

`benchmark/{2d,3d,3d_lung}_memory.py` are the peak-RAM counterparts of the timing
scripts (same datasets, same three-way maskel/skimage/VesselVio comparison), reporting
peak RSS, mask size, and foreground pixel/voxel count per sample. They write both a
printed table and a CSV to `results/` (gitignored, regenerable).

**Why they don't reuse the timing scripts' single-process loop.**
`resource.getrusage(RUSAGE_SELF).ru_maxrss` is a monotonic high-water mark for the
whole process — it never decreases. Timing all three methods back-to-back in one
process (as `*_comparison.py` does) would let an earlier, larger-footprint method's
peak leak into every later method's reported number. `benchmark/_memory_worker.py`
runs exactly one (sample, method) pair per subprocess invocation, so each method's
peak RSS is attributable only to that method. The three `*_memory.py` scripts just
loop over samples and shell out to it once per method.

## Notebooks

`analysis/HRF_Feature_Analysis.ipynb` and `analysis/HRF_Prediction.ipynb` process all 45 HRF samples through `maskel.pipeline.analyze_binary_image()` and explore/predict phenotype from the resulting features. Both need `HRF/` present and insert the repo root onto `sys.path` in their first cell to import local `hrf.py`.
