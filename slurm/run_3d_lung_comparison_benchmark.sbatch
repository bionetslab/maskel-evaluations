#!/bin/bash -l
#SBATCH --job-name=maskel-benchmark-3d-lung
#SBATCH --clusters=tinyfat
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Submit from the repo root: sbatch slurm/run_3d_lung_comparison_benchmark.sbatch
#
# Wall-clock timing comparison (maskel vs skimage vs VesselVio) on the VESSEL12 lung
# scans: 20 scans x 5 repeats x 3 methods. Exclusive node access is deliberate: these
# are timing comparisons, and a shared node would let other jobs skew the numbers via
# CPU contention. The committed results/3d_lung_runtime.csv totals ~2.35h of measured
# thinning time across all 100 rows; the budget above leaves headroom for slower
# nodes/contention on top of that.
#
# Requires beforehand:
#   - a VesselVio checkout, path set via VESSELVIO_PATH below or in the env
#
# VESSEL12 is auto-downloaded into data/Vessel12 on first run if not already present
# -- see benchmark/vessel12.py's ensure_vessel12(). tinyfat compute nodes have no
# direct internet egress (unlike the login node), so route through the NHR proxy in
# case that download is needed.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export VESSELVIO_PATH="${VESSELVIO_PATH:-$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)/VesselVio}"

# tinyfat nodes are 2x AMD EPYC 7502 = 64 physical cores/128 SMT threads; pin
# numba explicitly to the physical core count rather than letting it default
# to every visible thread (SMT siblings mostly add cache contention for this
# kind of CPU-bound kernel, not real throughput). maskel and VesselVio both
# parallelize through numba; scikit-image doesn't use it at all.
export NUMBA_NUM_THREADS=64
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"

echo "Host: $(hostname)   CPUs: $(nproc)   uv: $(uv --version)"
echo "VESSELVIO_PATH=${VESSELVIO_PATH}"

uv sync --extra notebooks

echo "=== 3D lung comparison (VESSEL12, auto-downloaded) ==="
uv run python benchmark/3d_lung_comparison.py
