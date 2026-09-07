#!/bin/bash -l
#SBATCH --job-name=maskel-benchmark-2d-HRF
#SBATCH --clusters=tinyfat
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Submit from the repo root: sbatch slurm/run_2d_HRF_comparison_benchmark.sbatch
#
# Wall-clock timing comparison (maskel vs skimage vs VesselVio) on the 45 HRF
# images. Exclusive node access is deliberate: these are timing comparisons,
# and a shared node would let other jobs skew the numbers via CPU contention.
# The README's own "finishes in minutes" note is the basis for the budget
# above; interactive/login-node runs have always been fast enough that this
# job is mostly about exclusive-node reproducibility, not raw runtime.
#
# Requires beforehand:
#   - a VesselVio checkout, path set via VESSELVIO_PATH below or in the env
#
# HRF is auto-downloaded into data/HRF on first run if not already present --
# see benchmark/hrf.py's ensure_hrf(). tinyfat compute nodes have no direct
# internet egress (unlike the login node), so route through the NHR proxy in
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

echo "=== 2D comparison (HRF) ==="
uv run python benchmark/2d_HRF_comparison.py
