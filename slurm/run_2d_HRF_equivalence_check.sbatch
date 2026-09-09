#!/bin/bash -l
#SBATCH --job-name=maskel-equivalence-2d-HRF
#SBATCH --clusters=tinyfat
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Submit from the repo root: sbatch slurm/run_2d_HRF_equivalence_check.sbatch
#
# Correctness check (maskel vs scikit-image, exact array equality), not a timing
# comparison - unlike run_2d_HRF_comparison_benchmark.sbatch, this doesn't need
# --exclusive: node contention from other jobs can't change whether two arrays
# are bit-identical. A modest shared allocation is enough.
#
# HRF is auto-downloaded into data/HRF on first run if not already present --
# see benchmark/hrf.py's ensure_hrf(). tinyfat compute nodes have no direct
# internet egress (unlike the login node), so route through the NHR proxy in
# case that download is needed.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export NUMBA_NUM_THREADS=16
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"

echo "Host: $(hostname)   CPUs: $(nproc)   uv: $(uv --version)"

uv sync --extra notebooks

echo "=== 2D equivalence check (HRF) ==="
uv run python benchmark/2d_HRF_equivalence.py
