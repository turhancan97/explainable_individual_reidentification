#!/usr/bin/env bash
# TEMPLATE: copy to env.sh (gitignored) and edit for the cluster you are on. The values
# below are the GMUM ones this work ran with; see MIGRATION.md in rdd-parallel-benchmark.
# Site configuration for the Slurm launchers (probe.sh, finetune.sh, probe-parallel-*.sh).
#
# Launchers are run from this repository root and load this file with
# `source "${EXREID_ROOT:-$PWD}/env.sh"`. Each value below is a default: export the
# variable before calling a launcher to override it for one submission. The array tasks
# inherit the exported values through Slurm's `--export=ALL`.
#
# Cross-repository paths are explicit here — nothing assumes that the other repositories
# are checked out next to (or inside) this one.

export EXREID_ROOT="${EXREID_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export RDD_BENCHMARK_ROOT="${RDD_BENCHMARK_ROOT:-/home/kubaty/rdd-parallel-benchmark}"
export LYNX_FINETUNING_ROOT="${LYNX_FINETUNING_ROOT:-/home/kubaty/lynx-finetuning-czechlynx}"

# Conda (GMUM): launchers call `activate_conda_env "$CONDA_ENV_EXREID"` (defined below).
export CONDA_SH="${CONDA_SH:-/shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh}"
export CONDA_ENV_EXREID="${CONDA_ENV_EXREID:-ex-reid}"

# Activate a conda environment and make sure its interpreter is the one on PATH — a
# submitting shell (e.g. a VS Code terminal) may carry another env's bin directory in
# front of PATH, and Slurm forwards that environment to the job (--export=ALL).
activate_conda_env() {
  source "${CONDA_SH}"
  conda activate "$1"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
}

# Feature caches written by train/probe.py (conf/probe.yaml reads this through
# ${oc.env:PROBE_CACHE_ROOT}). kargin's original cache root is not readable for us.
export PROBE_CACHE_ROOT="${PROBE_CACHE_ROOT:-/shared/results/kubaty/lynx/results}"

# Data and checkpoints. CHECKPOINTS_ROOT holds kargin's read-only fine-tuned checkpoints;
# checkpoints we train for the few-shot experiments live under FEWSHOT_ROOT.
export CZECHLYNX_DATA_ROOT="${CZECHLYNX_DATA_ROOT:-/shared/sets/datasets/vision/czechlynx}"
export CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${CZECHLYNX_DATA_ROOT}/checkpoints}"
export FEWSHOT_ROOT="${FEWSHOT_ROOT:-${CZECHLYNX_DATA_ROOT}/fewshot}"
