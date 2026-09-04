"""3D Lung Benchmark: maskel vs skimage vs VesselVio on VESSEL12 scans.

Runs each method directly on the raw binarized scan, with none of maskel's own
optional preprocessing (closing/fill_holes) applied first. This benchmark exists to
compare the three thinning *implementations* against each other; preprocessing is a
maskel-specific pipeline option, not something skimage/VesselVio have an equivalent
of, so leaving it out keeps the comparison to exactly what all three methods share.
An earlier version of this benchmark did apply preprocessing before timing - see git
history if you need to reproduce that configuration.

Masks are auto-downloaded from Zenodo into data/Vessel12 on first run - see
benchmark/vessel12.py's ensure_vessel12(). Writes results/3d_lung_runtime.csv.
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize as skimage_thin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vessel12 import Vessel12Dataset, ensure_vessel12

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

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "data" / "Vessel12"
_RESULTS = _HERE.parent / "results"


def warmup():
    small = np.zeros((10, 10, 10), dtype=np.uint8)
    small[2:8, 2:8, 2:8] = 1
    lee94_thin(small)


def print_row(name, lee_t, sk_lee_t, vv_t, speedup_sk=None, speedup_vv=None):
    if speedup_sk is None and speedup_vv is None:
        print(f"{name:<10} {lee_t:<16} {sk_lee_t:<16} {vv_t:<16}")
    elif all(isinstance(x, (int, float)) for x in (speedup_sk, speedup_vv)):
        print(
            f"{name:<10} {lee_t:<16.3f} {sk_lee_t:<16.3f} {vv_t:<16.3f} "
            f"{speedup_sk:<12.2f} {speedup_vv:<12.2f}"
        )
    else:
        print(
            f"{name:<10} {lee_t:<16} {sk_lee_t:<16} {vv_t:<16} "
            f"{speedup_sk!s:<12} {speedup_vv!s:<12}"
        )


def main():
    print("=" * 95)
    print("  VESSEL12 Lung Benchmark: maskel vs skimage vs VesselVio")
    print("  No preprocessing (raw binarized scan) - thinning implementations only")
    print("=" * 95)
    print()

    print("Warming up numba JIT...")
    warmup()
    print("JIT warmup done.\n")

    ensure_vessel12(_DATA)
    ds = Vessel12Dataset(_DATA)
    _RESULTS.mkdir(exist_ok=True)
    csv_path = _RESULTS / "3d_lung_runtime.csv"

    agg_lee, agg_sk, agg_vv = [], [], []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "repeat",
                "shape",
                "foreground_px",
                "foreground_pct",
                "maskel_time_s",
                "sk_lee_time_s",
                "vv_lee_time_s",
            ]
        )

        for proc, info in ds:
            name = info["name"]
            print(f"\n--- {name} ---")

            fg_px = int(np.count_nonzero(proc))
            fg_pct = 100 * fg_px / proc.size
            print(f"  shape={proc.shape}, foreground={fg_pct:.1f}%")

            print_row(
                "Run",
                "maskel(s)",
                "sk lee(s)",
                "vv lee(s)",
                "Spdup sk",
                "Spdup vv",
            )
            print("-" * 85)

            lee_times, sk_lee_times, vv_times = [], [], []

            for i in range(5):
                t0 = time.perf_counter()
                lee94_thin(proc)
                lee_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                skimage_thin(proc, method="lee")
                sk_lee_t = time.perf_counter() - t0

                vv_input = np.ascontiguousarray(np.pad(proc, 1).copy())
                t0 = time.perf_counter()
                vesselvio_lee94_thin(vv_input)
                vv_t = time.perf_counter() - t0

                lee_times.append(lee_t)
                sk_lee_times.append(sk_lee_t)
                vv_times.append(vv_t)

                writer.writerow(
                    [
                        name,
                        i,
                        proc.shape,
                        fg_px,
                        round(fg_pct, 3),
                        round(lee_t, 6),
                        round(sk_lee_t, 6),
                        round(vv_t, 6),
                    ]
                )

                speedup_sk = sk_lee_t / lee_t if lee_t > 0 else float("inf")
                speedup_vv = vv_t / lee_t if lee_t > 0 else float("inf")
                print_row(i, lee_t, sk_lee_t, vv_t, speedup_sk, speedup_vv)

            print("-" * 85)

            lt = np.median(lee_times)
            skt = np.median(sk_lee_times)
            vvt = np.median(vv_times)
            print_row("MEDIAN", lt, skt, vvt, skt / lt, vvt / lt)

            agg_lee.extend(lee_times)
            agg_sk.extend(sk_lee_times)
            agg_vv.extend(vv_times)

    print(f"\nWrote {csv_path}")

    print("\n" + "=" * 95)
    print("  AGGREGATE (all runs, all scans)")
    print("=" * 95)
    print_row("Stat", "maskel(s)", "sk lee(s)", "vv lee(s)", "Spdup sk", "Spdup vv")
    print("-" * 85)
    stats = [
        ("TOTAL", sum(agg_lee), sum(agg_sk), sum(agg_vv)),
        ("MEAN", np.mean(agg_lee), np.mean(agg_sk), np.mean(agg_vv)),
        ("MEDIAN", np.median(agg_lee), np.median(agg_sk), np.median(agg_vv)),
    ]
    for name, lt, skt, vvt in stats:
        print_row(name, lt, skt, vvt, skt / lt, vvt / lt)
    print("=" * 95)


if __name__ == "__main__":
    main()
