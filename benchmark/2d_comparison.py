"""2D Benchmark: maskel vs skimage vs VesselVio on HRF dataset"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize as skimage_thin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrf import HRFDataset, preprocess_segmentation

VESSELVIO_PATH = os.environ.get("VESSELVIO_PATH")
if not VESSELVIO_PATH:
    raise SystemExit(
        "Set VESSELVIO_PATH to a local VesselVio checkout to run this benchmark "
        "(https://github.com/JacobBumgarner/VesselVio)."
    )
sys.path.append(VESSELVIO_PATH)
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
    ds = HRFDataset("HRF")
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

    for i in range(len(ds)):
        _, seg, mask, info = ds.load_sample(i)
        cleaned = preprocess_segmentation(seg, mask)

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


if __name__ == "__main__":
    main()
