#!/bin/bash -l
#SBATCH --job-name=maskel-mem-2d
#SBATCH --clusters=tinyfat
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Submit from the repo root: sbatch slurm/run_2d_memory_benchmark.sbatch
#
# Peak-RAM comparison (maskel vs skimage vs VesselVio) on the HRF dataset. Runs
# benchmark/2d_memory.py, which spawns one subprocess per (sample, method) pair via
# benchmark/_memory_worker.py -- see that file's docstring for why. Exclusive node
# access matches the timing job: not strictly needed for memory measurements, but
# keeps the wall-time column (also recorded) comparable to the timing benchmark.
#
# Requires beforehand:
#   - HRF/ dataset at the repo root (see README.md)
#   - a VesselVio checkout, path set via VESSELVIO_PATH below or in the env

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export VESSELVIO_PATH="${VESSELVIO_PATH:-$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)/VesselVio}"

echo "Host: $(hostname)   CPUs: $(nproc)   uv: $(uv --version)"
echo "VESSELVIO_PATH=${VESSELVIO_PATH}"

uv sync --extra dev --extra notebooks

echo "=== 2D memory comparison (HRF) ==="
uv run python benchmark/2d_memory.py
