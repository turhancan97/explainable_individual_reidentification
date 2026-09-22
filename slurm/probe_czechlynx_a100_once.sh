#!/usr/bin/env bash
# One-off: finish the CzechLynx few-shot probes on the A100 partition (rtx4090 is busy).
#
# Submits, through probe-fewshot-wildlife.sh, only the variants that still lack a completed
# run (both retrieval settings; right now that is RDD-LightGlue default @ reduced gallery 1/4)
# and then the collector job with a dependency on them. Everything else is untouched.
#
# Run from this repository root:   bash slurm/probe_czechlynx_a100_once.sh
# Before that, cancel a still-pending rtx4090 resubmission of the same variant (otherwise it
# is computed twice):              scancel <jobid>   (e.g. the pending 509490)
# Options: PROBE_QOS (quick), FEWSHOT_FRACTIONS ("0.125 0.25 0.5 1.0"), PROBE_CANDIDATE_K (50).
set -euo pipefail
source "${EXREID_ROOT:-$PWD}/env.sh"
cd "${EXREID_ROOT}"
export HF_HUB_OFFLINE=1                                  # backbone from the local HF cache (token expired)
export PROBE_SBATCH_ARGS="-p dgxa100 --qos=${PROBE_QOS:-quick}"   # 8xA100 node, per-user 6 GPU / 60 CPU / 384 GB
read -r -a fractions <<< "${FEWSHOT_FRACTIONS:-0.125 0.25 0.5 1.0}"

probe_jobs=()
for flag in "" "--full-gallery"; do
  echo "== CzechLynx probes on dgxa100, ${flag:-reduced gallery}"
  # shellcheck disable=SC2086
  output=$(bash probe-fewshot-wildlife.sh CzechLynx "${fractions[@]}" ${flag})
  echo "${output}"
  read -r -a ids <<< "$(sed -n 's/^PROBE_JOBS=//p' <<< "${output}" | tail -n 1)"
  probe_jobs+=("${ids[@]}")
done
if (( ${#probe_jobs[@]} == 0 )); then
  echo "nothing missing: every variant already has a completed run -> collecting now"
  collector=$(sbatch --parsable --job-name=fs-collect-CzechLynx slurm/fewshot_collect.sh CzechLynx)
else
  dependency="afterany:$(IFS=:; echo "${probe_jobs[*]}")"
  collector=$(sbatch --parsable --dependency="${dependency}" --job-name=fs-collect-CzechLynx slurm/fewshot_collect.sh CzechLynx)
  echo "probe arrays on dgxa100: ${probe_jobs[*]}"
fi
echo "collector job: ${collector} -> ${EXREID_ROOT}/reports/fewshot/CzechLynx/"
