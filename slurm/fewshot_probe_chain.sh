#!/usr/bin/env bash
#SBATCH --job-name=fs-probe-chain
#SBATCH --partition=cpu
#SBATCH --qos=quick
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fewshot/%x-%j.out
#SBATCH --error=logs/fewshot/%x-%j.err
#
# Waiter job of the few-shot pipeline: runs once the fine-tuning jobs of a dataset are
# finished (it is submitted with --dependency on them), submits the probe arrays for the
# reduced- and the full-gallery setting and then the collector job that waits for them.
# Run from this repository root:
#   sbatch [--dependency=afterany:<train jobs>] slurm/fewshot_probe_chain.sh <animal> <fraction> [<fraction> ...]
# environment: everything probe-fewshot-wildlife.sh understands (PROBE_SBATCH_ARGS,
# PROBE_CANDIDATE_K, FEWSHOT_SEED, ...), FEWSHOT_GALLERIES ("reduced full").
set -euo pipefail
source "${EXREID_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh"
cd "${EXREID_ROOT}"
activate_conda_env "${CONDA_ENV_EXREID}"
animal=${1:?usage: fewshot_probe_chain.sh <animal> <fraction> [<fraction> ...]}
shift
(( $# > 0 )) || { echo "no fractions given" >&2; exit 1; }
fractions=("$@")
mkdir -p logs/fewshot

probe_jobs=()
for gallery in ${FEWSHOT_GALLERIES:-reduced full}; do
  flag=""; [[ "${gallery}" == full ]] && flag="--full-gallery"
  echo "== ${animal}: probes, ${gallery} gallery (fractions: ${fractions[*]})"
  output=$(bash probe-fewshot-wildlife.sh "${animal}" "${fractions[@]}" ${flag})
  echo "${output}"
  read -r -a ids <<< "$(sed -n 's/^PROBE_JOBS=//p' <<< "${output}" | tail -n 1)"
  probe_jobs+=("${ids[@]}")
done
echo "probe arrays: ${probe_jobs[*]:-none}"

dependency=""
if (( ${#probe_jobs[@]} > 0 )); then dependency="--dependency=afterany:$(IFS=:; echo "${probe_jobs[*]}")"; fi
# shellcheck disable=SC2086
collector=$(sbatch --parsable ${dependency} --job-name="fs-collect-${animal}" slurm/fewshot_collect.sh "${animal}")
echo "collector job: ${collector}${dependency:+ (${dependency})}"
echo "results will be written to ${EXREID_ROOT}/reports/fewshot/${animal}/"
