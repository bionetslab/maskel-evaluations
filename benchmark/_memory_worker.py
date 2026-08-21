"""Memory-measurement worker: runs ONE thinning method on ONE sample, in its own process.

Not meant to be run standalone for real use -- benchmark/*_memory.py invoke it via
subprocess, one process per (sample, method) pair.

Why a fresh process per call: resource.getrusage(RUSAGE_SELF).ru_maxrss is a monotonic
high-water mark for the entire life of the process. Looping over methods in one
long-lived process (like the *_comparison.py timing scripts do) would let an earlier,
larger-footprint method's peak leak into every later method's reported number, even if
that later method actually used less. A fresh process per (sample, method) is the only
way to attribute peak RSS to the method that actually caused it.

Emits one JSON line on stdout: {"baseline_kb": ..., "peak_kb": ..., "wall_time_s": ...}
- baseline_kb: peak RSS right before the timed call (imports + JIT warmup + input
  already loaded/preprocessed in memory).
- peak_kb: peak RSS right after the timed call.
- wall_time_s: matches the timing window used by benchmark/*_comparison.py exactly
  (e.g. VesselVio's required pad+copy is excluded from wall_time_s but still runs
  before peak_kb is read, so its memory cost lands in the delta, not in wall_time_s).
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def _load_hrf(index: int) -> np.ndarray:
    from hrf import HRFDataset, preprocess_segmentation

    ds = HRFDataset(os.environ.get("HRF_PATH", str(_HERE.parent / "HRF")))
    _, seg, mask, _ = ds.load_sample(index)
    return preprocess_segmentation(seg, mask)


def _load_vessap() -> np.ndarray:
    import tifffile

    volume_path = os.environ.get(
        "VESSAP_VOLUME_PATH", str(_HERE.parent / "data" / "vessap_preprocessed.tif")
    )
    return tifffile.imread(volume_path).astype(np.uint8)


def _load_vessel12(scan_name: str) -> np.ndarray:
    import itk

    from maskel.pipeline import preprocess_binary

    mhd_path = _HERE.parent / "tests" / "lung_masks" / f"{scan_name}.mhd"
    vol = np.asarray(itk.imread(str(mhd_path)))
    binary = (vol > 0).astype(np.uint8)
    return preprocess_binary(
        binary, closing_iterations=1, fill_holes=True, max_hole_size=100
    )


def _run_maskel(vol: np.ndarray) -> tuple[int, float]:
    from maskel.thin import lee94_thin

    small = np.zeros((10,) * vol.ndim, dtype=np.uint8)
    small[tuple(slice(2, 8) for _ in range(vol.ndim))] = 1
    lee94_thin(small)  # numba JIT warmup, excluded from the measured call

    baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    lee94_thin(vol)
    wall_time_s = time.perf_counter() - t0
    return baseline_kb, wall_time_s


def _run_skimage(vol: np.ndarray, method: str) -> tuple[int, float]:
    from skimage.morphology import skeletonize

    baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    skeletonize(vol > 0, method=method)
    wall_time_s = time.perf_counter() - t0
    return baseline_kb, wall_time_s


def _run_vesselvio(vol: np.ndarray) -> tuple[int, float]:
    vesselvio_path = os.environ["VESSELVIO_PATH"]
    sys.path.append(vesselvio_path)
    import skimage.morphology

    if not hasattr(skimage.morphology, "skeletonize_3d"):
        # VesselVio pins scikit-image==0.18.1; skeletonize_3d was folded into
        # skeletonize (n-dim aware) and removed in modern scikit-image.
        skimage.morphology.skeletonize_3d = skimage.morphology.skeletonize
    from library.lee94 import skeletonize as vesselvio_lee94_thin

    baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # VesselVio's Lee94 requires 3D uint8, padded by 1, C-contiguous. Not part of
    # wall_time_s (matches benchmark/*_comparison.py), but its memory cost is real
    # and lands in the delta since baseline_kb was already captured above.
    padded = vol if vol.ndim == 3 else vol[np.newaxis, ...]
    vv_input = np.ascontiguousarray(np.pad(padded, 1).copy())
    t0 = time.perf_counter()
    vesselvio_lee94_thin(vv_input)
    wall_time_s = time.perf_counter() - t0
    return baseline_kb, wall_time_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["hrf", "vessap", "vessel12"])
    parser.add_argument(
        "--method", required=True, choices=["maskel", "sk_zhang", "sk_lee", "vv_lee"]
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="HRF sample index or VESSEL12 scan name (e.g. VESSEL12_01); unused for vessap",
    )
    args = parser.parse_args()

    if args.dataset == "hrf":
        vol = _load_hrf(int(args.sample))
    elif args.dataset == "vessap":
        vol = _load_vessap()
    else:
        vol = _load_vessel12(args.sample)

    if args.method == "maskel":
        baseline_kb, wall_time_s = _run_maskel(vol)
    elif args.method in ("sk_zhang", "sk_lee"):
        baseline_kb, wall_time_s = _run_skimage(vol, args.method.split("_", 1)[1])
    else:
        baseline_kb, wall_time_s = _run_vesselvio(vol)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"baseline_kb": baseline_kb, "peak_kb": peak_kb, "wall_time_s": wall_time_s}))


if __name__ == "__main__":
    main()
