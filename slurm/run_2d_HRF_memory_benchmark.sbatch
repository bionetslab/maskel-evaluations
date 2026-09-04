#!/bin/bash -l
#SBATCH --job-name=maskel-mem-2d-HRF
#SBATCH --clusters=tinyfat
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Submit from the repo root: sbatch slurm/run_2d_HRF_memory_benchmark.sbatch
#
# Peak-RAM comparison (maskel vs skimage vs VesselVio) on the HRF dataset. Runs
# benchmark/2d_HRF_memory.py, which spawns one subprocess per (sample, method) pair
# via benchmark/_memory_worker.py -- see that file's docstring for why. Exclusive
# node access matches the timing job: not strictly needed for memory measurements,
# but keeps the wall-time column (also recorded) comparable to the timing benchmark.
#
# Requires beforehand:
#   - a VesselVio checkout, path set via VESSELVIO_PATH below or in the env
#
# The HRF manual segmentations are auto-downloaded into data/HRF on first run (see
# benchmark/hrf.py's ensure_hrf()) if not already present -- tinyfat compute nodes
# have no direct internet egress (unlike the login node), so route through the NHR
# proxy in case that download is needed.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export VESSELVIO_PATH="${VESSELVIO_PATH:-$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)/VesselVio}"
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"

echo "Host: $(hostname)   CPUs: $(nproc)   uv: $(uv --version)"
echo "VESSELVIO_PATH=${VESSELVIO_PATH}"

uv sync --extra notebooks

echo "=== 2D memory comparison (HRF) ==="
uv run python benchmark/2d_HRF_memory.py
