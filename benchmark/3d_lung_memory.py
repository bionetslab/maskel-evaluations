"""3D Lung Memory Benchmark: peak RAM of maskel vs skimage vs VesselVio on VESSEL12 scans.

Masks must already be present in tests/lung_masks/ -- run benchmark/3d_lung_comparison.py
first if they aren't (it auto-downloads them from Zenodo); this script doesn't fetch
them itself.

Runs each method directly on the raw binarized scan, with none of maskel's own
optional preprocessing (closing/fill_holes) applied first. This benchmark exists to
compare the three thinning *implementations* against each other; preprocessing is a
maskel-specific pipeline option, not something skimage/VesselVio have an equivalent
of, so leaving it out keeps the comparison to exactly what all three methods share.
An earlier version of this benchmark did apply preprocessing before thinning - see
git history if you need to reproduce that configuration.

Each (scan, method) pair runs in benchmark/_memory_worker.py's own subprocess -- see
that file's docstring for why, and for why the scan is loaded here rather than inside
the worker (even without preprocessing, itk.imread's own overhead is large enough -
several hundred MB, per manual RSS instrumentation - to risk swamping a method's own
delta_mb the same way preprocess_binary used to).
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import itk
import numpy as np

VESSELVIO_PATH = os.environ.get("VESSELVIO_PATH")
if not VESSELVIO_PATH:
    raise SystemExit(
        "Set VESSELVIO_PATH to a local VesselVio checkout to run this benchmark "
        "(https://github.com/JacobBumgarner/VesselVio)."
    )

_HERE = Path(__file__).resolve().parent
_WORKER = _HERE / "_memory_worker.py"
_RESULTS = _HERE.parent / "results"
LUNG_DIR = _HERE.parent / "tests" / "lung_masks"

METHODS = ["maskel", "sk_lee", "vv_lee"]
METHOD_LABELS = {"maskel": "maskel", "sk_lee": "sk lee", "vv_lee": "vv lee"}


def measure(array_path: Path, method: str) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "--array-path",
            str(array_path),
            "--method",
            method,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def kb_to_mb(kb: float) -> float:
    return kb / 1024


def main():
    scans = sorted(LUNG_DIR.glob("VESSEL12_*.mhd"))
    if not scans:
        raise SystemExit(
            f"No VESSEL12_*.mhd files found in {LUNG_DIR}. Run "
            "benchmark/3d_lung_comparison.py first to download them."
        )

    _RESULTS.mkdir(exist_ok=True)
    csv_path = _RESULTS / "3d_lung_memory.csv"

    header = (
        f"{'Scan':<14} {'shape':<16} {'mask_MB':<9} {'fg_px':<12} {'fg_%':<7} "
        + " ".join(f"{METHOD_LABELS[m] + ' peakMB':<14}" for m in METHODS)
    )
    print(header)
    print("-" * len(header))

    with open(csv_path, "w", newline="") as f, tempfile.TemporaryDirectory() as tmp_dir:
        writer = csv.writer(f)
        writer.writerow(
            ["scan", "shape", "mask_bytes", "foreground_px", "foreground_pct"]
            + [f"{m}_baseline_mb" for m in METHODS]
            + [f"{m}_peak_mb" for m in METHODS]
            + [f"{m}_delta_mb" for m in METHODS]
            + [f"{m}_wall_time_s" for m in METHODS]
        )

        for mhd_path in scans:
            name = mhd_path.stem
            vol = np.asarray(itk.imread(str(mhd_path)))
            proc = (vol > 0).astype(np.uint8)
            fg_px = int(np.count_nonzero(proc))
            fg_pct = 100 * fg_px / proc.size
            mask_bytes = proc.nbytes

            array_path = Path(tmp_dir) / f"{name}.npy"
            np.save(array_path, proc)
            results = {m: measure(array_path, m) for m in METHODS}
            array_path.unlink()

            row = (
                f"{name:<14} {str(proc.shape):<16} {mask_bytes / 1e6:<9.2f} "
                f"{fg_px:<12} {fg_pct:<7.1f} "
            )
            row += " ".join(f"{kb_to_mb(results[m]['peak_kb']):<14.1f}" for m in METHODS)
            print(row)

            writer.writerow(
                [name, proc.shape, mask_bytes, fg_px, round(fg_pct, 3)]
                + [round(kb_to_mb(results[m]["baseline_kb"]), 2) for m in METHODS]
                + [round(kb_to_mb(results[m]["peak_kb"]), 2) for m in METHODS]
                + [
                    round(kb_to_mb(results[m]["peak_kb"] - results[m]["baseline_kb"]), 2)
                    for m in METHODS
                ]
                + [round(results[m]["wall_time_s"], 3) for m in METHODS]
            )

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
