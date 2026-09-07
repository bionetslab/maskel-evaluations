"""Memory-measurement worker: runs ONE thinning method on ONE already-preprocessed
array, in its own process.

Not meant to be run standalone for real use -- benchmark/*_memory.py invoke it via
subprocess, one process per (sample, method) pair, passing the path to a .npy file
that the caller already loaded and preprocessed.

Why VmRSS polling, not resource.getrusage().ru_maxrss. ru_maxrss is a per-process
high-water mark that Linux seeds from the PARENT's RSS at fork() time (dup_mm()
copies the parent's current RSS into the child's hiwater_rss), and that inherited
value survives execve() into this program - verified empirically: a subprocess
spawned from a parent holding ~420MB of its own arrays reported ru_maxrss=430MB
despite the child itself only ever using ~12MB. That's exactly the kind of
contamination benchmark/*_memory.py's calling scripts are vulnerable to, since they
hold each sample's array in memory across a loop before spawning this worker.
VmRSS in /proc/self/status is a live current-usage snapshot instead, immune to that
inheritance - but being a snapshot rather than a running maximum, we have to poll it
ourselves (_PeakRssSampler) to catch the peak reached during a call, since there's
no kernel-provided "peak just during this window" value the way ru_maxrss pretends
to offer (incorrectly, per the above) for the whole process.

Why a fresh process per (sample, method) pair, still. baseline_kb/peak_kb are now a
local, windowed measurement (peak observed during THIS call minus VmRSS immediately
before it), so a later method's delta would actually be correct even if measured in
the same long-lived process as an earlier method - the windowing itself cancels out
whatever the earlier method left behind. Separate processes remain worthwhile anyway
for the same reason `*_comparison.py` avoids sharing engine handles/caches across
methods otherwise: it's the simplest way to guarantee zero cross-method interference
(fd state, allocator arena reuse, C-extension thread-local caches, etc.), not because
the measurement itself demands it the way it did under ru_maxrss.

Emits one JSON line on stdout: {"baseline_kb": ..., "peak_kb": ..., "wall_time_s": ...}
- baseline_kb: VmRSS right before the timed call (imports + JIT warmup + the
  preprocessed array loaded from disk).
- peak_kb: highest VmRSS observed while the timed call was running.
- wall_time_s: matches the timing window used by benchmark/*_comparison.py exactly
  (e.g. VesselVio's required pad+copy is excluded from wall_time_s but still runs
  inside the sampled window, so its memory cost lands in the delta, not in wall_time_s).
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

# Pin the incidental single-threaded-baseline libraries before numpy/scipy
# import (and therefore before any BLAS gets a chance to spin up its own
# thread pool) - purely for run-to-run reproducibility, since scikit-image's
# skeletonize doesn't call into BLAS for its own work anyway. NUMBA_NUM_THREADS
# is deliberately left alone here (setdefault, not assignment): the caller
# (a Slurm job script, typically) is expected to pin it explicitly to the
# node's physical core count, since numba is what maskel and VesselVio both
# actually parallelize through - this worker only reports whatever value
# is in effect (see main()), not force one.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def _vmrss_kb() -> int:
    """Current resident set size, in KB - see module docstring for why this is
    used instead of resource.getrusage().ru_maxrss."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


class _PeakRssSampler:
    """Polls VmRSS on a background thread for the duration of a `with` block,
    tracking the highest value observed - see module docstring for why VmRSS
    needs active polling rather than a single before/after read (a call that
    spikes and frees memory before we get to check again would otherwise be
    invisible). Relies on the timed call releasing the GIL during its own
    compiled (numba/C) work - true for numba, scipy, skimage, and VesselVio's
    C extensions for the bulk of their execution - so this thread actually
    gets scheduled while the main thread is inside the timed call.
    """

    _POLL_INTERVAL_S = 0.01

    def __init__(self):
        self._peak_kb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            self._peak_kb = max(self._peak_kb, _vmrss_kb())
            self._stop.wait(self._POLL_INTERVAL_S)

    def __enter__(self):
        self._peak_kb = _vmrss_kb()
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join()
        self._peak_kb = max(self._peak_kb, _vmrss_kb())

    @property
    def peak_kb(self) -> int:
        return self._peak_kb


def _run_maskel(vol: np.ndarray) -> tuple[int, int, float]:
    from maskel.thin import lee94_thin

    small = np.zeros((10,) * vol.ndim, dtype=np.uint8)
    small[tuple(slice(2, 8) for _ in range(vol.ndim))] = 1
    lee94_thin(small)  # numba JIT warmup, excluded from the measured call

    baseline_kb = _vmrss_kb()
    with _PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        lee94_thin(vol)
        wall_time_s = time.perf_counter() - t0
    return baseline_kb, sampler.peak_kb, wall_time_s


def _run_skimage(vol: np.ndarray, method: str) -> tuple[int, int, float]:
    from skimage.morphology import skeletonize

    baseline_kb = _vmrss_kb()
    with _PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        skeletonize(vol > 0, method=method)
        wall_time_s = time.perf_counter() - t0
    return baseline_kb, sampler.peak_kb, wall_time_s


def _run_vesselvio(vol: np.ndarray) -> tuple[int, int, float]:
    vesselvio_path = os.environ["VESSELVIO_PATH"]
    sys.path.append(vesselvio_path)
    import skimage.morphology

    if not hasattr(skimage.morphology, "skeletonize_3d"):
        # VesselVio pins scikit-image==0.18.1; skeletonize_3d was folded into
        # skeletonize (n-dim aware) and removed in modern scikit-image.
        skimage.morphology.skeletonize_3d = skimage.morphology.skeletonize
    from library.lee94 import skeletonize as vesselvio_lee94_thin

    # numba JIT warmup, excluded from the measured call - VesselVio's own
    # lee94.py is @njit(cache=True) throughout, same as maskel's, so without
    # this it pays first-call compile/cache-load time inside the timed
    # window on every single (sample, method) subprocess, while maskel's
    # equivalent overhead is always excluded by its own warmup above.
    small = np.zeros((10, 10, 10), dtype=np.uint8)
    small[2:8, 2:8, 2:8] = 1
    vesselvio_lee94_thin(np.ascontiguousarray(np.pad(small, 1)))

    baseline_kb = _vmrss_kb()
    padded = vol if vol.ndim == 3 else vol[np.newaxis, ...]
    with _PeakRssSampler() as sampler:
        # VesselVio's Lee94 requires 3D uint8, padded by 1, C-contiguous. Not part
        # of wall_time_s (matches benchmark/*_comparison.py), but its memory cost
        # is real and lands in the delta since it runs inside the sampled window.
        vv_input = np.ascontiguousarray(np.pad(padded, 1).copy())
        t0 = time.perf_counter()
        vesselvio_lee94_thin(vv_input)
        wall_time_s = time.perf_counter() - t0
    return baseline_kb, sampler.peak_kb, wall_time_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--array-path",
        required=True,
        help="Path to a .npy file holding the already-preprocessed input array",
    )
    parser.add_argument(
        "--method", required=True, choices=["maskel", "sk_zhang", "sk_lee", "vv_lee"]
    )
    args = parser.parse_args()

    import numba

    print(
        f"NUMBA_NUM_THREADS in effect: {numba.get_num_threads()}",
        file=sys.stderr,
    )

    vol = np.load(args.array_path)

    if args.method == "maskel":
        baseline_kb, peak_kb, wall_time_s = _run_maskel(vol)
    elif args.method in ("sk_zhang", "sk_lee"):
        baseline_kb, peak_kb, wall_time_s = _run_skimage(vol, args.method.split("_", 1)[1])
    else:
        baseline_kb, peak_kb, wall_time_s = _run_vesselvio(vol)

    print(json.dumps({"baseline_kb": baseline_kb, "peak_kb": peak_kb, "wall_time_s": wall_time_s}))


if __name__ == "__main__":
    main()
