#!/bin/bash -l
#SBATCH --job-name=maskel-equivalence-3d-lung
#SBATCH --clusters=tinyfat
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#
# Submit from the repo root: sbatch slurm/run_3d_lung_equivalence_check.sbatch
#
# Correctness check (maskel vs scikit-image, exact array equality), not a timing
# comparison - unlike run_3d_lung_comparison_benchmark.sbatch, this doesn't need
# --exclusive: node contention from other jobs can't change whether two arrays
# are bit-identical. A modest shared allocation is enough. Each scan is thinned
# once per method (no repeats, no VesselVio), so the budget above is generous
# relative to the committed per-call runtimes in results/3d_lung_runtime.csv.
#
# VESSEL12 is auto-downloaded into data/Vessel12 on first run if not already present
# -- see benchmark/vessel12.py's ensure_vessel12(). tinyfat compute nodes have no
# direct internet egress (unlike the login node), so route through the NHR proxy in
# case that download is needed.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export NUMBA_NUM_THREADS=16
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"

echo "Host: $(hostname)   CPUs: $(nproc)   uv: $(uv --version)"

uv sync --extra notebooks

echo "=== 3D equivalence check (VESSEL12, auto-downloaded) ==="
uv run python benchmark/3d_lung_equivalence.py
