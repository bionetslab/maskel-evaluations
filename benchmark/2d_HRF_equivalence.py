"""2D Equivalence Check: does maskel's Lee94 output match scikit-image's, exactly?

maskel's own test suite (tests/test_thin_2d.py) already regression-tests this against
a single scikit-image demo image (data.horse()). This script instead checks every one
of the 45 HRF images actually used for the 2D runtime benchmark
(benchmark/2d_HRF_comparison.py), since that's the claim the paper actually makes:
bit-identical output on the benchmarked datasets, not just on one canned sample.

This is a correctness check, not a timing benchmark - no JIT warmup or thread pinning
needed, since a Numba compile pause doesn't change what gets computed. Writes
results/2d_HRF_equivalence.csv.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize as skimage_thin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrf import HRFDataset, ensure_hrf

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "data" / "HRF"
_RESULTS = _HERE.parent / "results"

from maskel.thin import lee94_thin


def main():
    ensure_hrf(_DATA)
    ds = HRFDataset(_DATA)
    _RESULTS.mkdir(exist_ok=True)
    csv_path = _RESULTS / "2d_HRF_equivalence.csv"

    print(f"{'Sample':<10} {'Shape':<16} {'Match':<8} {'Mismatched px':<16}")
    print("-" * 55)

    n_exact = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "shape",
                "foreground_px",
                "exact_match",
                "n_mismatched",
                "pct_of_foreground",
            ]
        )

        for i in range(len(ds)):
            _, seg, _mask, info = ds.load_sample(i)
            cleaned = seg
            fg_px = int(np.count_nonzero(cleaned))

            maskel_skel = lee94_thin(cleaned)
            scikit_skel = skimage_thin(cleaned > 0, method="lee").astype(np.uint8)

            exact_match = bool(np.array_equal(maskel_skel, scikit_skel))
            n_mismatched = int(np.count_nonzero(maskel_skel != scikit_skel))
            pct_of_fg = 100 * n_mismatched / fg_px if fg_px > 0 else 0.0

            if exact_match:
                n_exact += 1

            writer.writerow(
                [
                    info["name"],
                    cleaned.shape,
                    fg_px,
                    exact_match,
                    n_mismatched,
                    round(pct_of_fg, 6),
                ]
            )

            print(
                f"{info['name']:<10} {str(cleaned.shape):<16} {str(exact_match):<8} {n_mismatched:<16}"
            )

    print("-" * 55)
    print(f"\n{n_exact}/{len(ds)} HRF images were bit-identical to scikit-image's Lee94 output.")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
