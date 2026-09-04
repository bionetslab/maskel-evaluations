"""2D Benchmark: maskel vs skimage vs VesselVio on HRF dataset.

Runs each method directly on the raw manual segmentation (no preprocessing) - see
module docstring in benchmark/3d_lung_comparison.py for why these benchmarks compare
the thinning implementations in isolation rather than mixing in maskel's own optional
preprocessing step.

The HRF manual segmentations are auto-downloaded into data/HRF on first run - see
benchmark/hrf.py's ensure_hrf(). Writes results/2d_HRF_runtime.csv.
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize as skimage_thin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrf import HRFDataset, ensure_hrf

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "data" / "HRF"
_RESULTS = _HERE.parent / "results"

VESSELVIO_PATH = os.environ.get("VESSELVIO_PATH")
if not VESSELVIO_PATH:
    raise SystemExit(
        "Set VESSELVIO_PATH to a local VesselVio checkout to run this benchmark "
        "(https://github.com/JacobBumgarner/VesselVio)."
    )
sys.path.append(VESSELVIO_PATH)
import skimage.morphology

if not hasattr(skimage.morphology, "skeletonize_3d"):
    # VesselVio pins scikit-image==0.18.1; skeletonize_3d was folded into
    # skeletonize (n-dim aware) and removed in modern scikit-image.
    skimage.morphology.skeletonize_3d = skimage.morphology.skeletonize
from library.lee94 import skeletonize as vesselvio_lee94_thin
from maskel.thin import lee94_thin


def warmup():
    small = np.zeros((10, 10), dtype=np.uint8)
    small[2:8, 2:8] = 1
    lee94_thin(small)


def print_row(
    name,
    lee_t,
    sk_zhang_t,
    sk_lee_t,
    vv_t,
    speedup_zhang=None,
    speedup_lee=None,
    speedup_vv=None,
):
    if all(x is None for x in (speedup_zhang, speedup_lee, speedup_vv)):
        print(f"{name:<10} {lee_t:<12} {sk_zhang_t:<12} {sk_lee_t:<12} {vv_t:<12}")
    elif all(
        isinstance(x, (int, float)) for x in (speedup_zhang, speedup_lee, speedup_vv)
    ):
        print(
            f"{name:<10} {lee_t:<12.3f} {sk_zhang_t:<12.3f} {sk_lee_t:<12.3f} {vv_t:<12.3f} {speedup_zhang:<12.2f} {speedup_lee:<12.2f} {speedup_vv:<12.2f}"
        )
    else:
        print(
            f"{name:<10} {lee_t:<12} {sk_zhang_t:<12} {sk_lee_t:<12} {vv_t:<12} {speedup_zhang!s:<12} {speedup_lee!s:<12} {speedup_vv!s:<12}"
        )


def main():
    ensure_hrf(_DATA)
    ds = HRFDataset(_DATA)
    _RESULTS.mkdir(exist_ok=True)
    csv_path = _RESULTS / "2d_HRF_runtime.csv"

    print("Warming up numba JIT...")
    warmup()
    print("JIT warmup done.\n")

    print_row(
        "Sample",
        "maskel(s)",
        "sk zhang(s)",
        "sk lee(s)",
        "vv lee(s)",
        "Spdup zhang",
        "Spdup lee",
        "Spdup vv",
    )
    print("-" * 90)

    lee_times, sk_zhang_times, sk_lee_times, vv_times = [], [], [], []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "shape",
                "foreground_px",
                "foreground_pct",
                "maskel_time_s",
                "sk_zhang_time_s",
                "sk_lee_time_s",
                "vv_lee_time_s",
            ]
        )

        for i in range(len(ds)):
            _, seg, _mask, info = ds.load_sample(i)
            cleaned = seg
            fg_px = int(np.count_nonzero(cleaned))
            fg_pct = 100 * fg_px / cleaned.size

            t0 = time.perf_counter()
            lee94_thin(cleaned)
            lee_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            skimage_thin(cleaned > 0, method="zhang")
            sk_zhang_t = time.perf_counter() - t0

            t0 = time.perf_counter()
            skimage_thin(cleaned > 0, method="lee")
            sk_lee_t = time.perf_counter() - t0

            # VesselVio Lee94 expects 3D uint8 padded by 1 on all sides
            vv_input = np.ascontiguousarray(np.pad(cleaned[np.newaxis, ...], 1).copy())
            t0 = time.perf_counter()
            vesselvio_lee94_thin(vv_input)
            vv_t = time.perf_counter() - t0

            lee_times.append(lee_t)
            sk_zhang_times.append(sk_zhang_t)
            sk_lee_times.append(sk_lee_t)
            vv_times.append(vv_t)

            writer.writerow(
                [
                    info["name"],
                    cleaned.shape,
                    fg_px,
                    round(fg_pct, 3),
                    round(lee_t, 6),
                    round(sk_zhang_t, 6),
                    round(sk_lee_t, 6),
                    round(vv_t, 6),
                ]
            )

            speedup_zhang = sk_zhang_t / lee_t if lee_t > 0 else float("inf")
            speedup_lee = sk_lee_t / lee_t if lee_t > 0 else float("inf")
            speedup_vv = vv_t / lee_t if lee_t > 0 else float("inf")
            print_row(
                info["name"],
                lee_t,
                sk_zhang_t,
                sk_lee_t,
                vv_t,
                speedup_zhang,
                speedup_lee,
                speedup_vv,
            )

    print("-" * 90)

    stats = [
        (
            "TOTAL",
            sum(lee_times),
            sum(sk_zhang_times),
            sum(sk_lee_times),
            sum(vv_times),
        ),
        (
            "MEAN",
            np.mean(lee_times),
            np.mean(sk_zhang_times),
            np.mean(sk_lee_times),
            np.mean(vv_times),
        ),
        (
            "MEDIAN",
            np.median(lee_times),
            np.median(sk_zhang_times),
            np.median(sk_lee_times),
            np.median(vv_times),
        ),
    ]

    for name, lt, szt, slt, vvt in stats:
        speedup_zhang = szt / lt
        speedup_lee = slt / lt
        speedup_vv = vvt / lt
        print_row(name, lt, szt, slt, vvt, speedup_zhang, speedup_lee, speedup_vv)

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
