#!/usr/bin/env bash
# Probe benchmark for the few-shot views of a WildlifeReID dataset (or CzechLynx: pass the
# animal name `CzechLynx`, which selects probe-parallel-czechlynx.sh and its metadata).
#
# Two retrieval settings:
#   reduced gallery (default)  the kept training images of the view are the database
#                              (profile split column split_frac<f>_seed<s>: train/unused/test);
#   --full-gallery             the whole training split is the database (split column `split`)
#                              while the fine-tuned checkpoint comes from the view. Only the
#                              fine-tuned variants depend on the fraction there; the default
#                              checkpoints are submitted once (PROBE_FULL_DEFAULTS=0 skips them).
#
# For every requested fraction one probe-parallel-wildlife.sh array is submitted through the
# launcher's PROBE_PROFILE / PROBE_VARIANTS / PROBE_CANDIDATE_K / PROBE_SBATCH_ARGS overrides.
# The metadata copy metadata_fewshot/metadata_<animal>.csv and the checkpoints under
# $FEWSHOT_ROOT/checkpoints/<animal>/<protocol>/<view>/ are produced by
# rdd-parallel-benchmark's few-shot pipeline. The submitted array job ids are printed as
# `PROBE_JOBS=<id> <id> ...` on the last line.
#
# usage (from this repository root):
#   bash probe-fewshot-wildlife.sh CowDataset 0.125 0.25 0.5 1.0
#   bash probe-fewshot-wildlife.sh CowDataset 0.125 0.25 0.5 1.0 --full-gallery
#   bash probe-fewshot-wildlife.sh NyalaData 0.25 0.5 1.0 --dry-run
# environment:
#   FEWSHOT_SEED (0), FEWSHOT_PROTOCOL (legacy), FEWSHOT_EVAL_EPOCH (299)
#   PROBE_CANDIDATE_K ("50")            candidate budgets, space separated ("10 50 100 250" for the
#                                       accuracy-vs-k figures): one array per budget, so a budget
#                                       whose runs exist is skipped on its own
#   PROBE_FEWSHOT_VARIANTS              "method|matcher|label|path;..." replaces the variant table
#   PROBE_FEWSHOT_TAG                   experiment tag of the checkpoints (e.g. ep30 for a 30-epoch
#                                       run): directories <backend>[-<component>]-finetuned-<tag>,
#                                       labels custom[-<component>]-<tag>; empty = the untagged
#                                       300-epoch runs
#   PROBE_FEWSHOT_COMPONENTS            descriptor checkpoints of the view to probe when present
#                                       ("descriptor lg-descriptor descriptor-matcher"; "" = none):
#                                       <view>/rdd-descriptor-finetuned, rdd-lg-descriptor-finetuned
#                                       (RDD + LightGlue files of one epoch directory),
#                                       loma-descriptor-finetuned, loma-descriptor-matcher-finetuned;
#                                       submitted with the label custom-<component>
#   PROBE_SBATCH_ARGS ("-p rtx4090_batch --qos=batch --exclude=c11,c15,c22")   Slurm overrides
#                                       for the arrays (c11/c15: CUDA init fails for the ex-reid
#                                       torch build, c22: RTX 5090 unsupported by it; add
#                                       --dependency=... to chain after other jobs); the probe
#                                       tasks are short, so they go to the rtx4090 partition and
#                                       do not queue behind the fine-tuning jobs on dgxh100
#   MAX_CONCURRENT_JOBS (4)             array throttle
#   PROBE_FULL_DEFAULTS (1)             --full-gallery: also submit the default checkpoints
#   PROBE_SKIP_EXISTING (1)             leave out variants that already have a completed run
#                                       (experiments/probe, same split column and budget)
set -euo pipefail
source "${EXREID_ROOT:-$PWD}/env.sh"
cd "${EXREID_ROOT}"
activate_conda_env "${CONDA_ENV_EXREID}"   # reid.* imports (run discovery) and the manifest helper

animal=${1:?usage: $0 <animal> <fraction> [<fraction> ...] [--full-gallery] [--dry-run|--list-tasks]}
shift
fractions=()
passthrough=()
gallery=reduced
for arg in "$@"; do
  case "${arg}" in
    --dry-run|--list-tasks) passthrough+=("${arg}") ;;
    --full-gallery) gallery=full ;;
    --reduced-gallery) gallery=reduced ;;
    *) fractions+=("${arg}") ;;
  esac
done
(( ${#fractions[@]} > 0 )) || { echo "no fractions given" >&2; exit 1; }

seed=${FEWSHOT_SEED:-0}
protocol=${FEWSHOT_PROTOCOL:-legacy}
epoch=${FEWSHOT_EVAL_EPOCH:-299}
if [[ "${animal}" == CzechLynx ]]; then
  # CzechLynx: real (unmasked) images with the metadata mask, time-closed split, its own launcher
  dataset_name=CzechLynx_v2
  dataset_root=${CZECHLYNX_DATA_ROOT}/CzechLynx_v2
  metadata_file=metadata_fewshot/CzechLynxDataset-Metadata-Real.csv
  label_col=unique_name; no_background=false; image_variant=background; original_split_col=split-time_closed
  launcher=probe-parallel-czechlynx.sh
  profile_suffix=""
else
  dataset_name=WildlifeReID-10k
  dataset_root=${CZECHLYNX_DATA_ROOT}/WildlifeReID-10k
  metadata_file=metadata_fewshot/metadata_${animal}.csv
  label_col=identity; no_background=false; image_variant=no_background; original_split_col=split
  launcher=probe-parallel-wildlife.sh
  profile_suffix="|legacy|legacy|${epoch}|${epoch}"
fi
[[ -f "${dataset_root}/${metadata_file}" ]] || { echo "few-shot metadata not found: ${dataset_root}/${metadata_file} (run the few-shot view preparation first)" >&2; exit 1; }
header=$(head -n 1 "${dataset_root}/${metadata_file}" | tr -d "\r")
defaults="cosine|-|default|-;wildfusion|-|default|-;vismatch|loma|default|-;vismatch|rdd-lightglue|default|-"
jobs=()

read -r -a candidate_ks <<< "${PROBE_CANDIDATE_K:-50}"   # one submission per budget: see filter_missing

filter_missing() {  # filter_missing <split_col> <variants> <candidate_k> -> variants without a completed run
  if [[ "${PROBE_SKIP_EXISTING:-1}" == 0 ]]; then echo "${2}"; return 0; fi
  python3 scripts/fewshot_probe_missing.py --animal "${animal}" --split-col "${1}" \
    --candidate-k "${3}" --variants "${2}"
}

submit() {  # submit <profile> <variants> <candidate_k>
  local output
  output=$(PROBE_PROFILE="${1}" PROBE_VARIANTS="${2}" \
    PROBE_CANDIDATE_K="${3}" \
    PROBE_SBATCH_ARGS="${PROBE_SBATCH_ARGS:--p rtx4090_batch --qos=batch --exclude=c11,c15,c22}" \
    MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-4}" \
    RDD_CUSTOM_CHECKPOINT_PATH="${rdd_ckpt:-}" LOMA_CUSTOM_CHECKPOINT_PATH="${loma_ckpt:-}" \
    bash "${launcher}" "${passthrough[@]}")
  echo "${output}"
  local job
  job=$(sed -n 's/^Submitted batch job \([0-9]\+\).*/\1/p' <<< "${output}" | tail -n 1)
  [[ -n "${job}" ]] && jobs+=("${job}")
  return 0
}

for fraction in "${fractions[@]}"; do
  tag=$(python3 -c "f=float('${fraction}'); t=f'{f:.6f}'.rstrip('0').rstrip('.'); print(t if '.' in t else t+'.0')")
  view="frac${tag}-seed${seed}"
  split_col="split_frac${tag}_seed${seed}"
  [[ ",${header}," == *",${split_col},"* ]] || { echo "column ${split_col} missing in ${metadata_file}; prepare the ${view} view first" >&2; exit 1; }
  ckpt_root=${FEWSHOT_ROOT}/checkpoints/${animal}/${protocol}/${view}
  # epoch directories: the RDD trainer writes epoch_%02d, the LoMa trainer epoch_%03d
  rdd_epoch=$(printf 'epoch_%02d' "${epoch}"); loma_epoch=$(printf 'epoch_%03d' "${epoch}")
  tag=${PROBE_FEWSHOT_TAG:-}; suffix=${tag:+-${tag}}
  rdd_ckpt=${ckpt_root}/rdd-finetuned${suffix}/${rdd_epoch}/model.safetensors
  loma_ckpt=${ckpt_root}/loma-finetuned${suffix}/${loma_epoch}/model.safetensors
  finetuned=""
  if [[ -f "${rdd_ckpt}" ]]; then finetuned+="vismatch|rdd-lightglue|custom${suffix}|${rdd_ckpt};"; else echo "note: no RDD few-shot checkpoint for ${view} (${rdd_ckpt}); skipping its fine-tuned variant"; fi
  if [[ -f "${loma_ckpt}" ]]; then finetuned+="vismatch|loma|custom${suffix}|${loma_ckpt};"; fi
  # descriptor fine-tuning runs (czechlynx_train_job.sh with a training component): the RDD
  # descriptor file alone, the RDD + LightGlue epoch directory, or the LoMa file holding the
  # descriptor (and matcher) weights; the probe loads the components present in each
  for component in ${PROBE_FEWSHOT_COMPONENTS-descriptor lg-descriptor descriptor-matcher}; do
    case "${component}" in
      descriptor)
        [[ -f "${ckpt_root}/rdd-descriptor-finetuned${suffix}/${rdd_epoch}/model.safetensors" ]] && finetuned+="vismatch|rdd-lightglue|custom-descriptor${suffix}|${ckpt_root}/rdd-descriptor-finetuned${suffix}/${rdd_epoch}/model.safetensors;"
        [[ -f "${ckpt_root}/loma-descriptor-finetuned${suffix}/${loma_epoch}/model.safetensors" ]] && finetuned+="vismatch|loma|custom-descriptor${suffix}|${ckpt_root}/loma-descriptor-finetuned${suffix}/${loma_epoch}/model.safetensors;" ;;
      lg-descriptor)
        [[ -f "${ckpt_root}/rdd-lg-descriptor-finetuned${suffix}/${rdd_epoch}/model_1.safetensors" ]] && finetuned+="vismatch|rdd-lightglue|custom-lg-descriptor${suffix}|${ckpt_root}/rdd-lg-descriptor-finetuned${suffix}/${rdd_epoch};" ;;
      descriptor-matcher)
        [[ -f "${ckpt_root}/loma-descriptor-matcher-finetuned${suffix}/${loma_epoch}/model.safetensors" ]] && finetuned+="vismatch|loma|custom-descriptor-matcher${suffix}|${ckpt_root}/loma-descriptor-matcher-finetuned${suffix}/${loma_epoch}/model.safetensors;" ;;
      *) echo "unknown PROBE_FEWSHOT_COMPONENTS entry: ${component}" >&2; exit 1 ;;
    esac
  done
  finetuned=${finetuned%;}
  if [[ -n "${PROBE_FEWSHOT_VARIANTS:-}" ]]; then
    variants=${PROBE_FEWSHOT_VARIANTS}
  elif [[ "${gallery}" == reduced ]]; then
    variants="${defaults}${finetuned:+;${finetuned}}"
  else
    variants=${finetuned}
  fi
  if [[ "${gallery}" == reduced ]]; then
    profile="fewshot-${animal}-${view}|${dataset_name}|${animal}|${dataset_root}|${metadata_file}|${label_col}|mask|${no_background}|${image_variant}|${split_col}|train|test|100${profile_suffix}"
  else
    profile="fewshot-${animal}-${view}-fullgallery|${dataset_name}|${animal}|${dataset_root}|${metadata_file}|${label_col}|mask|${no_background}|${image_variant}|${original_split_col}|train|test|100${profile_suffix}"
  fi
  for candidate_k in "${candidate_ks[@]}"; do
    missing=$(filter_missing "$([[ "${gallery}" == reduced ]] && echo "${split_col}" || echo "${original_split_col}")" "${variants}" "${candidate_k}")
    if [[ -z "${missing}" ]]; then echo "${view} (k=${candidate_k}): nothing to probe in the ${gallery}-gallery setting (runs exist or no checkpoint)"; continue; fi
    echo "== ${animal} ${view} (${gallery} gallery, k=${candidate_k}): variants=${missing}"
    submit "${profile}" "${missing}" "${candidate_k}"
  done
done

if [[ "${gallery}" == full && "${PROBE_FULL_DEFAULTS:-1}" != 0 && -z "${PROBE_FEWSHOT_VARIANTS:-}" ]]; then
  rdd_ckpt=""; loma_ckpt=""
  profile="fewshot-${animal}-fullgallery-default|${dataset_name}|${animal}|${dataset_root}|${metadata_file}|${label_col}|mask|${no_background}|${image_variant}|${original_split_col}|train|test|100${profile_suffix}"
  for candidate_k in "${candidate_ks[@]}"; do
    missing=$(filter_missing "${original_split_col}" "${defaults}" "${candidate_k}")
    if [[ -n "${missing}" ]]; then
      echo "== ${animal} full gallery, default checkpoints (fraction independent), k=${candidate_k}: variants=${missing}"
      submit "${profile}" "${missing}" "${candidate_k}"
    else
      echo "== ${animal} full gallery, default checkpoints (k=${candidate_k}): runs exist"
    fi
  done
fi
echo "PROBE_JOBS=${jobs[*]:-}"
