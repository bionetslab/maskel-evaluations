#!/bin/bash -l
#SBATCH --job-name=maskel-benchmark
#SBATCH --clusters=tinyfat
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Submit from the repo root: sbatch slurm/run_benchmarks.sbatch
#
# Runs the maskel-evaluations benchmark suite (maskel vs skimage vs VesselVio)
# on a dedicated tinyfat node. Exclusive node access is deliberate: these are
# wall-clock timing comparisons, and a shared node would let other jobs skew
# the numbers via CPU contention.
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

echo "=== 2D comparison (HRF) ==="
uv run python benchmark/2d_comparison.py

echo "=== 3D comparison (VESSAP) ==="
uv run python benchmark/3d_comparison.py

echo "=== 3D lung comparison (VESSEL12, auto-downloaded) ==="
uv run python benchmark/3d_lung_comparison.py
