"""3D Equivalence Check: does maskel's Lee94 output match scikit-image's, exactly?

maskel's own test suite (tests/test_3d_skimage_comparison.py) already regression-tests
this against a single scikit-image demo volume (data.brain()). This script instead
checks every one of the 20 VESSEL12 scans actually used for the 3D runtime benchmark
(benchmark/3d_lung_comparison.py), since that's the claim the paper actually makes:
bit-identical output on the benchmarked datasets, not just on one canned sample.

This is a correctness check, not a timing benchmark - no JIT warmup or thread pinning
needed, since a Numba compile pause doesn't change what gets computed. Writes
results/3d_lung_equivalence.csv.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize as skimage_thin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vessel12 import Vessel12Dataset, ensure_vessel12

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "data" / "Vessel12"
_RESULTS = _HERE.parent / "results"

from maskel.thin import lee94_thin


def main():
    ensure_vessel12(_DATA)
    ds = Vessel12Dataset(_DATA)
    _RESULTS.mkdir(exist_ok=True)
    csv_path = _RESULTS / "3d_lung_equivalence.csv"

    print(f"{'Scan':<14} {'Shape':<20} {'Match':<8} {'Mismatched vx':<16}")
    print("-" * 60)

    n_exact = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "shape",
                "foreground_vx",
                "exact_match",
                "n_mismatched",
                "pct_of_foreground",
            ]
        )

        for proc, info in ds:
            fg_vx = int(np.count_nonzero(proc))

            maskel_skel = lee94_thin(proc)
            scikit_skel = skimage_thin(proc, method="lee").astype(np.uint8)

            exact_match = bool(np.array_equal(maskel_skel, scikit_skel))
            n_mismatched = int(np.count_nonzero(maskel_skel != scikit_skel))
            pct_of_fg = 100 * n_mismatched / fg_vx if fg_vx > 0 else 0.0

            if exact_match:
                n_exact += 1

            writer.writerow(
                [
                    info["name"],
                    proc.shape,
                    fg_vx,
                    exact_match,
                    n_mismatched,
                    round(pct_of_fg, 6),
                ]
            )

            print(
                f"{info['name']:<14} {str(proc.shape):<20} {str(exact_match):<8} {n_mismatched:<16}"
            )

    print("-" * 60)
    print(f"\n{n_exact}/{len(ds)} VESSEL12 scans were bit-identical to scikit-image's Lee94 output.")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
