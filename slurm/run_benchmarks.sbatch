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
#   - a VesselVio checkout, path set via VESSELVIO_PATH below or in the env
#
# HRF and VESSEL12 are both auto-downloaded (into data/HRF and data/Vessel12) on
# first run if not already present -- see benchmark/hrf.py's ensure_hrf() and
# benchmark/vessel12.py's ensure_vessel12().

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export VESSELVIO_PATH="${VESSELVIO_PATH:-$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)/VesselVio}"

# tinyfat compute nodes have no direct internet egress (unlike the login
# node) - route through the NHR proxy so ensure_hrf()/ensure_vessel12() can
# auto-download their datasets on first run.
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"

echo "Host: $(hostname)   CPUs: $(nproc)   uv: $(uv --version)"
echo "VESSELVIO_PATH=${VESSELVIO_PATH}"

uv sync --extra notebooks

echo "=== 2D comparison (HRF) ==="
uv run python benchmark/2d_HRF_comparison.py

echo "=== 3D lung comparison (VESSEL12, auto-downloaded) ==="
uv run python benchmark/3d_lung_comparison.py
