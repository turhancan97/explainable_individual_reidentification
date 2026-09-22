#!/usr/bin/env bash
#SBATCH --job-name=fs-collect
#SBATCH --partition=cpu
#SBATCH --qos=quick
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fewshot/%x-%j.out
#SBATCH --error=logs/fewshot/%x-%j.err
#
# Collect the few-shot results of one animal (probe runs of both gallery settings, view
# descriptions, benchmark-repository evaluations) into reports/fewshot/<animal>/.
# Run from this repository root: sbatch slurm/fewshot_collect.sh <animal>
set -euo pipefail
source "${EXREID_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh"
cd "${EXREID_ROOT}"
activate_conda_env "${CONDA_ENV_EXREID}"
animal=${1:?usage: fewshot_collect.sh <animal>}
python scripts/fewshot_results.py --animal "${animal}" --seed "${FEWSHOT_SEED:-0}" \
  --candidate-k "${PROBE_COLLECT_K:-50}" --metric top_1 --metric top_5 --metric balanced_top_1 --metric mAP_at_k
echo
cat "reports/fewshot/${animal}/fewshot_summary.md"
