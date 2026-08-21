"""3D Lung Benchmark: maskel vs skimage vs VesselVio on VESSEL12 scans.

Preprocessing (fill_holes max_hole_size=100 + 1 closing) done on-the-fly
before timing, not included in benchmark measurements.

Masks are auto-downloaded from Zenodo on first run.
"""

import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import itk
import numpy as np
from skimage.morphology import skeletonize as skimage_thin

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
from maskel.pipeline import preprocess_binary
from maskel.thin import lee94_thin

_HERE = Path(__file__).resolve().parent
LUNG_DIR = _HERE.parent / "tests" / "lung_masks"
_MASK_TGZ = LUNG_DIR / "VESSEL12_01-20_Lungmasks.tar.bz2"
_MASK_URL = (
    "https://zenodo.org/records/8055066/files/"
    "VESSEL12_01-20_Lungmasks.tar.bz2?download=1"
)


def ensure_masks():
    if LUNG_DIR.is_dir() and any(LUNG_DIR.glob("VESSEL12_*.mhd")):
        return
    LUNG_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading lung masks from Zenodo...")
    urllib.request.urlretrieve(_MASK_URL, _MASK_TGZ)
    print("Extracting...")
    with tarfile.open(_MASK_TGZ, "r:bz2") as tar:
        tar.extractall(path=LUNG_DIR)
    _MASK_TGZ.unlink()
    print("Done.")


def load_scan(mhd_path: Path) -> np.ndarray:
    return np.asarray(itk.imread(str(mhd_path)))


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
    print("  Preprocessing: fill_holes (max_hole_size=100) + 1 closing")
    print("=" * 95)
    print()

    print("Warming up numba JIT...")
    warmup()
    print("JIT warmup done.\n")

    ensure_masks()

    scans = sorted(LUNG_DIR.glob("VESSEL12_*.mhd"))

    agg_lee, agg_sk, agg_vv = [], [], []

    for mhd_path in scans:
        name = mhd_path.stem
        print(f"\n--- {name} ---")

        vol = load_scan(mhd_path)
        binary = (vol > 0).astype(np.uint8)
        proc = preprocess_binary(
            binary, closing_iterations=1, fill_holes=True, max_hole_size=100
        )
        fg_pct = 100 * proc.mean()
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
