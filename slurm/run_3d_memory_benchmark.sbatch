#!/bin/bash -l
#SBATCH --job-name=maskel-mem-3d
#SBATCH --clusters=tinyfat
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Submit from the repo root: sbatch slurm/run_3d_memory_benchmark.sbatch
#
# Peak-RAM comparison (maskel vs skimage vs VesselVio) on the VESSAP volume. Runs
# benchmark/3d_memory.py, which spawns one subprocess per method via
# benchmark/_memory_worker.py -- see that file's docstring for why.
#
# Requires beforehand:
#   - data/vessap_preprocessed.tif is already in the repo, no setup needed
#   - a VesselVio checkout, path set via VESSELVIO_PATH below or in the env

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export VESSELVIO_PATH="${VESSELVIO_PATH:-$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)/VesselVio}"

echo "Host: $(hostname)   CPUs: $(nproc)   uv: $(uv --version)"
echo "VESSELVIO_PATH=${VESSELVIO_PATH}"

uv sync --extra dev --extra notebooks

echo "=== 3D memory comparison (VESSAP) ==="
uv run python benchmark/3d_memory.py
