#!/bin/bash -l
#SBATCH --job-name=maskel-mem-3d-lung
#SBATCH --clusters=tinyfat
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Submit from the repo root: sbatch slurm/run_3d_lung_memory_benchmark.sbatch
#
# Peak-RAM comparison (maskel vs skimage vs VesselVio) on the VESSEL12 lung scans.
# Runs benchmark/3d_lung_memory.py, which spawns one subprocess per (scan, method)
# pair via benchmark/_memory_worker.py -- see that file's docstring for why. Time
# budget is a rough guess (20 scans x 3 methods x one call each, but each subprocess
# pays fresh import/JIT overhead); adjust if it proves too tight or too generous.
#
# Requires beforehand:
#   - data/Vessel12/VESSEL12_*.mhd/.raw already present (run
#     benchmark/3d_lung_comparison.py first if not -- it auto-downloads them from
#     Zenodo, which needs the proxy exports below on tinyfat compute nodes)
#   - a VesselVio checkout, path set via VESSELVIO_PATH below or in the env

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export VESSELVIO_PATH="${VESSELVIO_PATH:-$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)/VesselVio}"

# tinyfat compute nodes have no direct internet egress (unlike the login node) --
# harmless here since the masks are expected to already be present, but kept for
# parity with run_benchmarks.sbatch in case this ever needs to fetch them itself.
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"

echo "Host: $(hostname)   CPUs: $(nproc)   uv: $(uv --version)"
echo "VESSELVIO_PATH=${VESSELVIO_PATH}"

uv sync --extra notebooks

echo "=== 3D lung memory comparison (VESSEL12) ==="
uv run python benchmark/3d_lung_memory.py
