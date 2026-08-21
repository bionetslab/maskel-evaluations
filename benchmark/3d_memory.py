"""3D Memory Benchmark: peak RAM of maskel vs skimage vs VesselVio on the VESSAP volume.

Each method runs in benchmark/_memory_worker.py's own subprocess -- see that file's
docstring for why.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

VESSELVIO_PATH = os.environ.get("VESSELVIO_PATH")
if not VESSELVIO_PATH:
    raise SystemExit(
        "Set VESSELVIO_PATH to a local VesselVio checkout to run this benchmark "
        "(https://github.com/JacobBumgarner/VesselVio)."
    )

_HERE = Path(__file__).resolve().parent
_WORKER = _HERE / "_memory_worker.py"
_RESULTS = _HERE.parent / "results"

VOLUME_PATH = os.environ.get(
    "VESSAP_VOLUME_PATH", str(_HERE.parent / "data" / "vessap_preprocessed.tif")
)

METHODS = ["maskel", "sk_lee", "vv_lee"]
METHOD_LABELS = {"maskel": "maskel", "sk_lee": "sk lee", "vv_lee": "vv lee"}


def measure(method: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(_WORKER), "--dataset", "vessap", "--method", method],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def kb_to_mb(kb: float) -> float:
    return kb / 1024


def main():
    vol = tifffile.imread(VOLUME_PATH).astype(np.uint8)
    fg_px = int(np.count_nonzero(vol))
    fg_pct = 100 * fg_px / vol.size
    mask_bytes = vol.nbytes

    print(
        f"Volume shape={vol.shape}, mask={mask_bytes / 1e6:.2f} MB, "
        f"foreground={fg_px} px ({fg_pct:.1f}%)\n"
    )

    _RESULTS.mkdir(exist_ok=True)
    csv_path = _RESULTS / "3d_memory.csv"

    results = {m: measure(m) for m in METHODS}

    header = f"{'Method':<10} {'baseline_MB':<14} {'peak_MB':<14} {'delta_MB':<14} {'wall_s':<10}"
    print(header)
    print("-" * len(header))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shape", "mask_bytes", "foreground_px", "foreground_pct"])
        writer.writerow([vol.shape, mask_bytes, fg_px, round(fg_pct, 3)])
        writer.writerow([])
        writer.writerow(["method", "baseline_mb", "peak_mb", "delta_mb", "wall_time_s"])

        for m in METHODS:
            r = results[m]
            baseline_mb = kb_to_mb(r["baseline_kb"])
            peak_mb = kb_to_mb(r["peak_kb"])
            delta_mb = peak_mb - baseline_mb
            print(
                f"{METHOD_LABELS[m]:<10} {baseline_mb:<14.1f} {peak_mb:<14.1f} "
                f"{delta_mb:<14.1f} {r['wall_time_s']:<10.3f}"
            )
            writer.writerow(
                [m, round(baseline_mb, 2), round(peak_mb, 2), round(delta_mb, 2), round(r["wall_time_s"], 3)]
            )

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
