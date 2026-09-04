"""2D Memory Benchmark: peak RAM of maskel vs skimage vs VesselVio on HRF dataset.

Runs each method directly on the raw manual segmentation (no preprocessing) - see
module docstring in benchmark/3d_lung_memory.py for why these benchmarks compare the
thinning implementations in isolation rather than mixing in maskel's own optional
preprocessing step.

Each (sample, method) pair runs in benchmark/_memory_worker.py's own subprocess --
see that file's docstring for why (RUSAGE_SELF.ru_maxrss is a monotonic per-process
high-water mark, so this can't reuse the timing scripts' single-process loop) and
for why the sample is loaded here rather than inside the worker.

The HRF manual segmentations are auto-downloaded into data/HRF on first run - see
benchmark/hrf.py's ensure_hrf().
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrf import HRFDataset, ensure_hrf

VESSELVIO_PATH = os.environ.get("VESSELVIO_PATH")
if not VESSELVIO_PATH:
    raise SystemExit(
        "Set VESSELVIO_PATH to a local VesselVio checkout to run this benchmark "
        "(https://github.com/JacobBumgarner/VesselVio)."
    )

_HERE = Path(__file__).resolve().parent
_WORKER = _HERE / "_memory_worker.py"
_DATA = _HERE.parent / "data" / "HRF"
_RESULTS = _HERE.parent / "results"

METHODS = ["maskel", "sk_zhang", "sk_lee", "vv_lee"]
METHOD_LABELS = {"maskel": "maskel", "sk_zhang": "sk zhang", "sk_lee": "sk lee", "vv_lee": "vv lee"}


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
    ensure_hrf(_DATA)
    ds = HRFDataset(_DATA)
    _RESULTS.mkdir(exist_ok=True)
    csv_path = _RESULTS / "2d_HRF_memory.csv"

    header = (
        f"{'Sample':<10} {'shape':<14} {'mask_MB':<9} {'fg_px':<10} {'fg_%':<7} "
        + " ".join(f"{METHOD_LABELS[m] + ' peakMB':<14}" for m in METHODS)
    )
    print(header)
    print("-" * len(header))

    with open(csv_path, "w", newline="") as f, tempfile.TemporaryDirectory() as tmp_dir:
        writer = csv.writer(f)
        writer.writerow(
            ["sample", "shape", "mask_bytes", "foreground_px", "foreground_pct"]
            + [f"{m}_baseline_mb" for m in METHODS]
            + [f"{m}_peak_mb" for m in METHODS]
            + [f"{m}_delta_mb" for m in METHODS]
            + [f"{m}_wall_time_s" for m in METHODS]
        )

        for i in range(len(ds)):
            _, seg, _mask, info = ds.load_sample(i)
            cleaned = seg
            fg_px = int(np.count_nonzero(cleaned))
            fg_pct = 100 * fg_px / cleaned.size
            mask_bytes = cleaned.nbytes

            array_path = Path(tmp_dir) / f"{info['name']}.npy"
            np.save(array_path, cleaned)
            results = {m: measure(array_path, m) for m in METHODS}
            array_path.unlink()

            row = (
                f"{info['name']:<10} {str(cleaned.shape):<14} {mask_bytes / 1e6:<9.2f} "
                f"{fg_px:<10} {fg_pct:<7.1f} "
            )
            row += " ".join(f"{kb_to_mb(results[m]['peak_kb']):<14.1f}" for m in METHODS)
            print(row)

            writer.writerow(
                [info["name"], cleaned.shape, mask_bytes, fg_px, round(fg_pct, 3)]
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
