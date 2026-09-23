#!/bin/bash -l
#SBATCH -p dgx
#SBATCH --gpus=1
#SBATCH --qos=big
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=c22,c11,c13,c15,dgx1
#SBATCH --job-name=probe_wildlife
#SBATCH --time=24:00:00
#SBATCH --output=logs/parallel_run/%x_%A_%a.out
#SBATCH --error=logs/parallel_run/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-12}"
CANDIDATE_K_VALUES=(10)
# CANDIDATE_K_VALUES=(50 100 250 500 1000)

# Leave empty to derive paths from the single active dataset profile. Explicit
# overrides remain supported, but must belong to that profile's animal.
animal_name="${animal_name:-}" # optional validation override; the active profile supplies the animal
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-299}"
LOMA_CUSTOM_CHECKPOINT_PATH="${LOMA_CUSTOM_CHECKPOINT_PATH:-}"
RDD_CUSTOM_CHECKPOINT_PATH="${RDD_CUSTOM_CHECKPOINT_PATH:-}"
DESCRIPTOR_LOMA_CUSTOM_CHECKPOINT_PATH="${DESCRIPTOR_LOMA_CUSTOM_CHECKPOINT_PATH:-}"
DESCRIPTOR_RDD_CUSTOM_CHECKPOINT_PATH="${DESCRIPTOR_RDD_CUSTOM_CHECKPOINT_PATH:-}"

SCRIPT_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
mkdir -p "${SCRIPT_DIR}/logs/parallel_run"
LOG_ROOT="${PROBE_PARALLEL_LOG_ROOT:-${SCRIPT_DIR}/logs/parallel_run}"
CONFIG_FILE="${PROBE_PARALLEL_CONFIG:-${SCRIPT_DIR}/conf/probe.yaml}"
[[ -f "${CONFIG_FILE}" ]] || CONFIG_FILE="${SCRIPT_SOURCE_DIR}/conf/probe.yaml"
MANIFEST_HELPER="${SCRIPT_DIR}/scripts/probe_parallel_manifest.py"
# LAUNCHER_NAME="$(basename -- "${BASH_SOURCE[0]}")"
LAUNCHER_NAME="probe-parallel-wildlife.sh"
LAUNCHER_PATH="${SCRIPT_DIR}/${LAUNCHER_NAME}"

# profile|dataset|animal|root|metadata|label|mask|no_background|image_variant|split_col|database_split|query_split|calibration_size|loma_checkpoint_dir|rdd_checkpoint_dir|loma_epoch|rdd_epoch|descriptor_loma_checkpoint|descriptor_rdd_checkpoint|descriptor_owner|evaluation_animal
DATASET_PROFILES=(
    # "zindi|WildlifeReID-10k|ZindiTurtleRecall|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_no_background/metadata_ZindiTurtleRecall.csv|identity|mask|false|no_background|split|train|test|100||||"
    # Uncomment exactly one profile at a time.
    # "nyala|WildlifeReID-10k|NyalaData|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_no_background/metadata_NyalaData.csv|identity|mask|false|no_background|split|train|test|100"
    # "whaleshark|WildlifeReID-10k|WhaleSharkID|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_no_background/metadata_WhaleSharkID.csv|identity|mask|false|no_background|split|train|test|100"
    # "beluga|WildlifeReID-10k|BelugaID|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_no_background/metadata_BelugaID.csv|identity|mask|false|no_background|split|train|test|100"
    # New WildlifeReID-10k profiles use the official masked metadata. The final
    # four fields select the known checkpoint layout/epoch for each animal.
    # "atrw|WildlifeReID-10k|ATRW|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_ATRW.csv|identity|mask|false|no_background|split|train|test|100|legacy|legacy|299|299"
    # "giraffes|WildlifeReID-10k|Giraffes|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_Giraffes.csv|identity|mask|false|no_background|split|train|test|100|legacy|legacy|299|299"
    "leopardid2022|WildlifeReID-10k|LeopardID2022|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_LeopardID2022.csv|identity|mask|false|no_background|split|train|test|100|legacy|legacy|299|299"
    # "hyenaid2022|WildlifeReID-10k|HyenaID2022|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_HyenaID2022.csv|identity|mask|false|no_background|split|train|test|100|legacy|legacy|299|299"
    # "giraffezebraid|WildlifeReID-10k|GiraffeZebraID|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_GiraffeZebraID.csv|identity|mask|false|no_background|split|train|test|100|legacy|legacy|299|299"
    # "cowdataset|WildlifeReID-10k|CowDataset|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_CowDataset.csv|identity|mask|false|no_background|split|train|test|100|legacy|legacy|299|299"
    # "stripespotter|WildlifeReID-10k|StripeSpotter|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_StripeSpotter.csv|identity|mask|false|no_background|split|train|test|100|legacy|legacy|299|299"
    # "seastarreid2023|WildlifeReID-10k|SeaStarReID2023|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_SeaStarReID2023.csv|identity|mask|false|no_background|split|train|test|100|legacy|legacy|299|299"
)

# method|matcher|checkpoint_label|checkpoint_path|checkpoint_components|train_mode|class_weighting
# Legacy row spelling compatibility: linear_probe|-|default|-|classifier|weighted
# linear_probe|-|default|-|partial|weighted
# linear_probe|-|default|-|all|weighted
VARIANTS=(
    "cosine|-|default|-|-|-|-"
    # "wildfusion|-|default|-|-|-|-"
    # "local_lightglue|-|default|-|-|-|-"
    # "linear_probe|-|default|-|-|classifier|weighted"
    # "linear_probe|-|default|-|classifier|unweighted"
    # "linear_probe|-|default|-|partial|weighted"
    # "linear_probe|-|default|-|partial|unweighted"
    # "linear_probe|-|default|-|-|all|weighted"
    # "linear_probe|-|default|-|all|unweighted"
    # "efficient_probe|-|default|-|classifier|weighted"
    # "efficient_probe|-|default|-|classifier|unweighted"
    # "efficient_probe|-|default|-|partial|weighted"
    # "efficient_probe|-|default|-|partial|unweighted"
    # "efficient_probe|-|default|-|all|weighted"
    # "efficient_probe|-|default|-|all|unweighted"
    # "vismatch|loma|default|-|-|-|-"
    # "vismatch|loma|custom|${LOMA_CUSTOM_CHECKPOINT_PATH}|matcher_only|-|-"
    # "vismatch|loma|descriptor-fine-tuned|${DESCRIPTOR_LOMA_CUSTOM_CHECKPOINT_PATH}|descriptor_only|-|-"
    # "vismatch|rdd-lightglue|default|-|-|-|-"
    # "vismatch|rdd-lightglue|custom|${RDD_CUSTOM_CHECKPOINT_PATH}|matcher_only|-|-"
    # "vismatch|rdd-lightglue|descriptor-fine-tuned|${DESCRIPTOR_RDD_CUSTOM_CHECKPOINT_PATH}|descriptor_only|-|-"
)

die() { echo "${LAUNCHER_NAME}: $*" >&2; exit 1; }
sanitize_component() { local value="${1:-unknown}"; value="${value//[^a-zA-Z0-9_.-]/_}"; [[ -n "${value}" ]] || value=unknown; printf '%s' "${value:0:96}"; }
validate_positive_integer() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer; got '$2'"; }
validate_nonnegative_integer() { [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer; got '$2'"; }

if (( ${#DATASET_PROFILES[@]} != 1 )); then
    die "exactly one DATASET_PROFILES entry must be active; found ${#DATASET_PROFILES[@]}"
fi

IFS='|' read -r ACTIVE_PROFILE_ID ACTIVE_DATASET_NAME ACTIVE_ANIMAL ACTIVE_DATASET_ROOT ACTIVE_METADATA_FILE ACTIVE_LABEL_COL ACTIVE_MASK_COL ACTIVE_NO_BACKGROUND ACTIVE_IMAGE_VARIANT ACTIVE_SPLIT_COL ACTIVE_DATABASE_SPLIT ACTIVE_QUERY_SPLIT ACTIVE_CALIBRATION_SIZE ACTIVE_LOMA_CHECKPOINT_DIR ACTIVE_RDD_CHECKPOINT_DIR ACTIVE_LOMA_EPOCH ACTIVE_RDD_EPOCH ACTIVE_DESCRIPTOR_LOMA_CHECKPOINT ACTIVE_DESCRIPTOR_RDD_CHECKPOINT ACTIVE_DESCRIPTOR_OWNER ACTIVE_EVALUATION_ANIMAL <<< "${DATASET_PROFILES[0]}"
if [[ -n "${animal_name}" && "${animal_name}" != "${ACTIVE_ANIMAL}" ]]; then
    die "animal_name='${animal_name}' does not match the active profile animal '${ACTIVE_ANIMAL}'"
fi
animal_name="${animal_name:-${ACTIVE_ANIMAL}}"
ACTIVE_LOMA_CHECKPOINT_DIR="${ACTIVE_LOMA_CHECKPOINT_DIR:-legacy}"
ACTIVE_RDD_CHECKPOINT_DIR="${ACTIVE_RDD_CHECKPOINT_DIR:-legacy}"
ACTIVE_LOMA_EPOCH="${ACTIVE_LOMA_EPOCH:-${CHECKPOINT_EPOCH}}"
ACTIVE_RDD_EPOCH="${ACTIVE_RDD_EPOCH:-${CHECKPOINT_EPOCH}}"
LOMA_CUSTOM_CHECKPOINT_PATH="${LOMA_CUSTOM_CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/${animal_name}/loma-finetuned/${ACTIVE_LOMA_CHECKPOINT_DIR}/epoch_${ACTIVE_LOMA_EPOCH}/model.safetensors}"
RDD_CUSTOM_CHECKPOINT_PATH="${RDD_CUSTOM_CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/${animal_name}/rdd-finetuned/${ACTIVE_RDD_CHECKPOINT_DIR}/epoch_${ACTIVE_RDD_EPOCH}/model.safetensors}"

TASKS=()
for profile in "${DATASET_PROFILES[@]}"; do
    IFS='|' read -r PROFILE_ID DATASET_NAME ANIMAL DATASET_ROOT METADATA_FILE LABEL_COL MASK_COL NO_BACKGROUND IMAGE_VARIANT SPLIT_COL DATABASE_SPLIT_VALUE QUERY_SPLIT_VALUE CALIBRATION_SIZE PROFILE_LOMA_CHECKPOINT_DIR PROFILE_RDD_CHECKPOINT_DIR PROFILE_LOMA_EPOCH PROFILE_RDD_EPOCH PROFILE_DESCRIPTOR_LOMA_CHECKPOINT PROFILE_DESCRIPTOR_RDD_CHECKPOINT PROFILE_DESCRIPTOR_OWNER PROFILE_EVALUATION_ANIMAL <<< "${profile}"
    RDD_OWNER="${ANIMAL}"; RDD_PROFILE_CHECKPOINT="${RDD_CUSTOM_CHECKPOINT_PATH}"
    LOMA_OWNER="${ANIMAL}"; LOMA_PROFILE_CHECKPOINT="${LOMA_CUSTOM_CHECKPOINT_PATH}"
    PROFILE_DESCRIPTOR_LOMA_CHECKPOINT="${PROFILE_DESCRIPTOR_LOMA_CHECKPOINT:-${DESCRIPTOR_LOMA_CUSTOM_CHECKPOINT_PATH}}"
    PROFILE_DESCRIPTOR_RDD_CHECKPOINT="${PROFILE_DESCRIPTOR_RDD_CHECKPOINT:-${DESCRIPTOR_RDD_CUSTOM_CHECKPOINT_PATH}}"
    PROFILE_DESCRIPTOR_OWNER="${PROFILE_DESCRIPTOR_OWNER:-${ANIMAL}}"
    PROFILE_EVALUATION_ANIMAL="${PROFILE_EVALUATION_ANIMAL:-${ANIMAL}}"
    for candidate_k in "${CANDIDATE_K_VALUES[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            IFS='|' read -r METHOD MATCHER CHECKPOINT_LABEL CHECKPOINT_PATH CHECKPOINT_COMPONENTS TRAIN_MODE CLASS_WEIGHTING <<< "${variant}"
            if [[ ( "${METHOD}" == linear_probe || "${METHOD}" == efficient_probe ) && "${candidate_k}" != "${CANDIDATE_K_VALUES[0]}" ]]; then
                continue
            fi
            CHECKPOINT_OWNER=-; CHECKPOINT_COMPONENTS="${CHECKPOINT_COMPONENTS:--}"; LOMA_ARCH=-
            [[ -n "${TRAIN_MODE}" ]] || TRAIN_MODE=-
            [[ -n "${CLASS_WEIGHTING}" ]] || CLASS_WEIGHTING=-
            if [[ ( "${METHOD}" == linear_probe || "${METHOD}" == efficient_probe ) && "${TRAIN_MODE}" != classifier && "${TRAIN_MODE}" != partial && "${TRAIN_MODE}" != all ]]; then
                die "${METHOD} variant must use train_mode classifier, partial, or all"
            fi
            if [[ ( "${METHOD}" == linear_probe || "${METHOD}" == efficient_probe ) && "${CLASS_WEIGHTING}" != weighted && "${CLASS_WEIGHTING}" != unweighted ]]; then
                die "${METHOD} variant must specify class_weighting weighted or unweighted"
            fi
            if [[ "${METHOD}" != linear_probe && "${METHOD}" != efficient_probe && "${TRAIN_MODE}" != - ]]; then
                die "only classifier-probe variants may specify train_mode; got '${TRAIN_MODE}' for ${METHOD}"
            fi
            if [[ "${METHOD}" != linear_probe && "${METHOD}" != efficient_probe && "${CLASS_WEIGHTING}" != - ]]; then
                die "only classifier-probe variants may specify class_weighting; got '${CLASS_WEIGHTING}' for ${METHOD}"
            fi
            if [[ "${CHECKPOINT_LABEL}" == custom || "${CHECKPOINT_LABEL}" == descriptor-fine-tuned ]]; then
                if [[ "${MATCHER}" == loma ]]; then
                    CHECKPOINT_OWNER="${LOMA_OWNER}"; LOMA_ARCH=LoMa-B
                    if [[ "${CHECKPOINT_LABEL}" == descriptor-fine-tuned ]]; then CHECKPOINT_PATH="${PROFILE_DESCRIPTOR_LOMA_CHECKPOINT}"; CHECKPOINT_OWNER="${PROFILE_DESCRIPTOR_OWNER}"; else CHECKPOINT_PATH="${LOMA_PROFILE_CHECKPOINT}"; fi
                fi
                if [[ "${MATCHER}" == rdd-lightglue ]]; then
                    CHECKPOINT_OWNER="${RDD_OWNER}"
                    if [[ "${CHECKPOINT_LABEL}" == descriptor-fine-tuned ]]; then CHECKPOINT_PATH="${PROFILE_DESCRIPTOR_RDD_CHECKPOINT}"; CHECKPOINT_OWNER="${PROFILE_DESCRIPTOR_OWNER}"; else CHECKPOINT_PATH="${RDD_PROFILE_CHECKPOINT}"; fi
                fi
                [[ "${METHOD}" == vismatch ]] || die "custom checkpoint variants are only valid for vismatch"
                [[ "${CHECKPOINT_COMPONENTS}" == matcher_only || "${CHECKPOINT_COMPONENTS}" == descriptor_only || "${CHECKPOINT_COMPONENTS}" == full ]] || die "${CHECKPOINT_LABEL} must declare matcher_only, descriptor_only, or full"
                [[ -n "${CHECKPOINT_PATH}" && "${CHECKPOINT_PATH}" != "-" ]] || die "${PROFILE_ID} has no explicit ${MATCHER} custom checkpoint"
            elif [[ "${MATCHER}" == loma ]]; then
                LOMA_ARCH=LoMa-B
            fi
            TASKS+=("${PROFILE_ID}|${DATASET_NAME}|${ANIMAL}|${DATASET_ROOT}|${METADATA_FILE}|${LABEL_COL}|${MASK_COL}|${NO_BACKGROUND}|${IMAGE_VARIANT}|${SPLIT_COL}|${DATABASE_SPLIT_VALUE}|${QUERY_SPLIT_VALUE}|${CALIBRATION_SIZE}|${METHOD}|${MATCHER}|${CHECKPOINT_LABEL}|${CHECKPOINT_PATH}|${CHECKPOINT_OWNER}|${CHECKPOINT_COMPONENTS}|${LOMA_ARCH}|${TRAIN_MODE}|${CLASS_WEIGHTING}|${candidate_k}|${PROFILE_EVALUATION_ANIMAL}")
        done
    done
done

print_task() {
    local i="$1" t="$2"
    IFS='|' read -r profile dataset animal root metadata label mask no_background image_variant split_col db_split query_split calibration method matcher checkpoint_label checkpoint_path owner components loma_arch train_mode class_weighting candidate evaluation_animal <<< "$t"
    printf 'index=%s profile=%s dataset=%s evaluation_animal=%s candidate_k=%s method=%s matcher=%s train_mode=%s class_weighting=%s checkpoint=%s components=%s checkpoint_owner=%s path=%s\n' "$i" "$profile" "$dataset" "$evaluation_animal" "$candidate" "$method" "$matcher" "$train_mode" "$class_weighting" "$checkpoint_label" "$components" "$owner" "$checkpoint_path"
}

validate_positive_integer MAX_CONCURRENT_JOBS "${MAX_CONCURRENT_JOBS}"
for candidate_k in "${CANDIDATE_K_VALUES[@]}"; do validate_positive_integer candidate_k "${candidate_k}"; done
(( ${#DATASET_PROFILES[@]} > 0 )) || die DATASET_PROFILES-is-empty
(( ${#VARIANTS[@]} > 0 )) || die VARIANTS-is-empty

if [[ "${1:-}" == --list-tasks ]]; then
    for index in "${!TASKS[@]}"; do print_task "${index}" "${TASKS[index]}"; done
    exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LAST_TASK_INDEX=$((${#TASKS[@]} - 1)); ARRAY_SPEC="0-${LAST_TASK_INDEX}%${MAX_CONCURRENT_JOBS}"
    echo "Submitting ${#TASKS[@]} probe tasks with array throttle ${MAX_CONCURRENT_JOBS}."
    SUBMISSION_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    SUBMISSION_DIR="${LOG_ROOT}/submissions/${SUBMISSION_ID}"
    mkdir -p "${SUBMISSION_DIR}"
    TASK_TABLE_PATH="${SUBMISSION_DIR}/tasks.tsv"
    printf '%s\n' "${TASKS[@]}" > "${TASK_TABLE_PATH}"
    [[ -f "${MANIFEST_HELPER}" ]] || die "manifest helper does not exist: ${MANIFEST_HELPER}"
    MANIFEST_PATH="$(python "${MANIFEST_HELPER}" create --submission-dir "${SUBMISSION_DIR}" --submission-id "${SUBMISSION_ID}" --config-file "${CONFIG_FILE}" --task-file "${TASK_TABLE_PATH}" --launcher-path "${LAUNCHER_PATH}" --repository-dir "${SCRIPT_DIR}")"
    echo "Immutable submission manifest: ${MANIFEST_PATH}"
    if [[ "${PROBE_PARALLEL_DRY_RUN:-0}" == 1 || "${1:-}" == --dry-run ]]; then
        echo "Dry run: no Slurm array submitted."
        echo "sbatch --array=${ARRAY_SPEC} --export=ALL,PROBE_PARALLEL_MANIFEST=${MANIFEST_PATH} ${LAUNCHER_PATH}"
        for index in "${!TASKS[@]}"; do print_task "${index}" "${TASKS[index]}"; done
        exit 0
    fi
    exec sbatch --array="${ARRAY_SPEC}" --export="ALL,PROBE_PARALLEL_MANIFEST=${MANIFEST_PATH}" "${LAUNCHER_PATH}"
fi

TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
validate_nonnegative_integer SLURM_ARRAY_TASK_ID "${TASK_INDEX}"
if [[ -z "${PROBE_PARALLEL_MANIFEST:-}" ]]; then
    [[ "${PROBE_PARALLEL_DRY_RUN:-0}" == 1 || "${1:-}" == --dry-run ]] || die "PROBE_PARALLEL_MANIFEST is required for array tasks"
    (( TASK_INDEX < ${#TASKS[@]} )) || die "array task index ${TASK_INDEX} is outside 0..$((${#TASKS[@]} - 1))"
    CURRENT_TASK="${TASKS[${TASK_INDEX}]}"
    IFS='|' read -r PROFILE_ID DATASET_NAME ANIMAL DATASET_ROOT METADATA_FILE LABEL_COL MASK_COL NO_BACKGROUND IMAGE_VARIANT SPLIT_COL DATABASE_SPLIT_VALUE QUERY_SPLIT_VALUE CALIBRATION_SIZE METHOD MATCHER CHECKPOINT_LABEL CHECKPOINT_PATH CHECKPOINT_OWNER CHECKPOINT_COMPONENTS LOMA_ARCH TRAIN_MODE CLASS_WEIGHTING CANDIDATE_K EVALUATION_ANIMAL <<< "${CURRENT_TASK}"
    CHECKPOINT_SOURCE=default; [[ "${CHECKPOINT_LABEL}" == custom || "${CHECKPOINT_LABEL}" == descriptor-fine-tuned ]] && CHECKPOINT_SOURCE=custom
    if [[ "${CHECKPOINT_SOURCE}" == custom && ! -e "${CHECKPOINT_PATH}" ]]; then CHECKPOINT_DISPLAY="${MATCHER}"; [[ "${MATCHER}" == rdd-lightglue ]] && CHECKPOINT_DISPLAY="RDD-LightGlue"; [[ "${MATCHER}" == loma ]] && CHECKPOINT_DISPLAY="LoMa"; die "${CHECKPOINT_DISPLAY} custom checkpoint does not exist: ${CHECKPOINT_PATH}"; fi
    CONFIG_SNAPSHOT_PATH="${CONFIG_FILE}"; SUBMISSION_ID=local-dry-run; CHECKPOINT_SHA256=
else
    [[ -f "${MANIFEST_HELPER}" ]] || die "manifest helper does not exist: ${MANIFEST_HELPER}"
    eval "$(python "${MANIFEST_HELPER}" emit-shell --manifest "${PROBE_PARALLEL_MANIFEST}" --index "${TASK_INDEX}")"
    if ! VALIDATION_OUTPUT="$(python "${MANIFEST_HELPER}" validate --manifest "${PROBE_PARALLEL_MANIFEST}" --index "${TASK_INDEX}" 2>&1)"; then
        LOG_DATASET="$(sanitize_component "${DATASET_NAME}")"; LOG_ANIMAL="$(sanitize_component "${ANIMAL}")"
        ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"; TASK_SLUG="${METHOD}-${MATCHER}"; [[ "${METHOD}" == linear_probe || "${METHOD}" == efficient_probe ]] && TASK_SLUG="${TASK_SLUG}-${TRAIN_MODE}-${CLASS_WEIGHTING}"; TASK_SLUG="$(sanitize_component "${TASK_SLUG}")"; TASK_CHECKPOINT="$(sanitize_component "${CHECKPOINT_LABEL}")"
        TASK_LOG_DIR="${LOG_ROOT}/${LOG_DATASET}/${LOG_ANIMAL}/job-${ARRAY_JOB_ID}"; TASK_STEM="task-$(printf '%03d' "${TASK_INDEX}")__${TASK_SLUG}__${TASK_CHECKPOINT}__k${CANDIDATE_K}"
        TASK_OUT_PATH="${TASK_LOG_DIR}/${TASK_STEM}.out"; TASK_ERR_PATH="${TASK_LOG_DIR}/${TASK_STEM}.err"; TASK_COMBINED_PATH="${TASK_LOG_DIR}/${TASK_STEM}.combined.log"; TASK_METADATA_PATH="${TASK_LOG_DIR}/${TASK_STEM}.json"; mkdir -p "${TASK_LOG_DIR}"
        printf '[launcher] immutable task validation failed: %s\n' "${VALIDATION_OUTPUT}" | tee -a "${TASK_ERR_PATH}" "${TASK_COMBINED_PATH}" >&2
        python scripts/probe_log_metadata.py init --path "${TASK_METADATA_PATH}" --job-id "${ARRAY_JOB_ID}" --task-id "${TASK_INDEX}" --dataset "${LOG_DATASET}" --animal "${LOG_ANIMAL}" --method "${METHOD}" --matcher "${MATCHER}" --train-mode "${TRAIN_MODE}" --class-weighting "${CLASS_WEIGHTING}" --checkpoint "${CHECKPOINT_LABEL}" --checkpoint-path "${CHECKPOINT_PATH}" --candidate-k "${CANDIDATE_K}" --command "validation-only: no probe execution" --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --stdout-path "${TASK_OUT_PATH}" --stderr-path "${TASK_ERR_PATH}" --combined-path "${TASK_COMBINED_PATH}" --submission-id "${SUBMISSION_ID}" --manifest-path "${PROBE_PARALLEL_MANIFEST}" --profile-id "${PROFILE_ID}" --checkpoint-owner "${CHECKPOINT_OWNER}" --checkpoint-sha256 "${CHECKPOINT_SHA256}" --validation-status failed --status failed
        python scripts/probe_log_metadata.py update --path "${TASK_METADATA_PATH}" --status failed --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --validation-status failed --validation-error "${VALIDATION_OUTPUT}" --error-file "${TASK_ERR_PATH}" || true
        python scripts/summarize_logs.py --logs-root "${LOG_ROOT}" --write-index --quiet || true
        if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then scancel "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" || true; elif [[ -n "${SLURM_JOB_ID:-}" ]]; then scancel "${SLURM_JOB_ID}" || true; fi
        exit 1
    fi
fi

LOG_DATASET="$(sanitize_component "${DATASET_NAME}")"; LOG_ANIMAL="$(sanitize_component "${ANIMAL}")"
IFS='|' read -r _ _ _ _ _ _ _ _ _ _ _ _ _ METHOD MATCHER CHECKPOINT_LABEL CHECKPOINT_PATH CHECKPOINT_OWNER CHECKPOINT_COMPONENTS LOMA_ARCH TRAIN_MODE CLASS_WEIGHTING CANDIDATE_K EVALUATION_ANIMAL <<< "${CURRENT_TASK:-${PROFILE_ID}|${DATASET_NAME}|${ANIMAL}|${DATASET_ROOT}|${METADATA_FILE}|${LABEL_COL}|${MASK_COL}|${NO_BACKGROUND}|${IMAGE_VARIANT}|${SPLIT_COL}|${DATABASE_SPLIT_VALUE}|${QUERY_SPLIT_VALUE}|${CALIBRATION_SIZE}|${METHOD}|${MATCHER}|${CHECKPOINT_LABEL}|${CHECKPOINT_PATH}|${CHECKPOINT_OWNER}|${CHECKPOINT_COMPONENTS}|${LOMA_ARCH}|${TRAIN_MODE}|${CLASS_WEIGHTING}|${CANDIDATE_K}|${ANIMAL}}"
CURRENT_TASK="${PROFILE_ID}|${DATASET_NAME}|${ANIMAL}|${DATASET_ROOT}|${METADATA_FILE}|${LABEL_COL}|${MASK_COL}|${NO_BACKGROUND}|${IMAGE_VARIANT}|${SPLIT_COL}|${DATABASE_SPLIT_VALUE}|${QUERY_SPLIT_VALUE}|${CALIBRATION_SIZE}|${METHOD}|${MATCHER}|${CHECKPOINT_LABEL}|${CHECKPOINT_PATH}|${CHECKPOINT_OWNER}|${CHECKPOINT_COMPONENTS}|${LOMA_ARCH}|${TRAIN_MODE}|${CLASS_WEIGHTING}|${CANDIDATE_K}|${EVALUATION_ANIMAL:-${ANIMAL}}"
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-default}"

PROBE_ARGS=(--config-dir "$(dirname -- "${CONFIG_SNAPSHOT_PATH}")" --config-name probe
    "dataset.name=${DATASET_NAME}" "dataset.animal=${ANIMAL}" "dataset.root=${DATASET_ROOT}" "dataset.metadata_file=${METADATA_FILE}"
    "dataset.label_col=${LABEL_COL}" "dataset.mask_col=${MASK_COL}" "dataset.no_background=${NO_BACKGROUND}" "dataset.image_variant=${IMAGE_VARIANT}"
    "dataset.split_col=${SPLIT_COL}" "dataset.database_split_value=${DATABASE_SPLIT_VALUE}" "dataset.query_split_value=${QUERY_SPLIT_VALUE}" "dataset.calibration_size=${CALIBRATION_SIZE}"
    "benchmark.method=${METHOD}" "benchmark.candidate_k=${CANDIDATE_K}")
if [[ "${METHOD}" == linear_probe || "${METHOD}" == efficient_probe ]]; then
    PROBE_ARGS+=("benchmark.methods.${METHOD}.train_mode=${TRAIN_MODE}")
    [[ "${CLASS_WEIGHTING}" == weighted ]] && PROBE_ARGS+=("benchmark.methods.${METHOD}.class_weighting=inverse_frequency")
    [[ "${CLASS_WEIGHTING}" == unweighted ]] && PROBE_ARGS+=("benchmark.methods.${METHOD}.class_weighting=none")
fi
if [[ "${METHOD}" == vismatch ]]; then
    PROBE_ARGS+=("benchmark.methods.vismatch.matcher=${MATCHER}")
    [[ "${MATCHER}" == loma ]] && PROBE_ARGS+=("benchmark.methods.vismatch.loma_arch=${LOMA_ARCH}")
    if [[ "${CHECKPOINT_SOURCE}" == custom ]]; then PROBE_ARGS+=("benchmark.methods.vismatch.checkpoint_source=custom" "benchmark.methods.vismatch.checkpoint_path=${CHECKPOINT_PATH}" "benchmark.methods.vismatch.checkpoint_components=${CHECKPOINT_COMPONENTS}" "benchmark.methods.vismatch.checkpoint_owner=${CHECKPOINT_OWNER}" "benchmark.methods.vismatch.evaluation_animal=${EVALUATION_ANIMAL}"); else PROBE_ARGS+=(benchmark.methods.vismatch.checkpoint_source=default); fi
fi

printf 'Starting probe array task %s\n' "${TASK_INDEX}"; print_task "${TASK_INDEX}" "${CURRENT_TASK:-${TASKS[${TASK_INDEX}]}}"
printf 'Submission ID: %s\nManifest: %s\nConfig snapshot: %s\nCheckpoint owner: %s\nCheckpoint SHA256: %s\n' "${SUBMISSION_ID}" "${PROBE_PARALLEL_MANIFEST:-local compatibility dry-run}" "${CONFIG_SNAPSHOT_PATH}" "${CHECKPOINT_OWNER:-}" "${CHECKPOINT_SHA256:-}"
printf 'python train/probe.py'; printf ' %q' "${PROBE_ARGS[@]}"; printf '\n'
if [[ "${PROBE_PARALLEL_DRY_RUN:-0}" == 1 || "${1:-}" == --dry-run ]]; then exit 0; fi

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"; TASK_SLUG="${METHOD}"; [[ "${MATCHER}" != - ]] && TASK_SLUG="${TASK_SLUG}-${MATCHER}"; [[ "${METHOD}" == linear_probe || "${METHOD}" == efficient_probe ]] && TASK_SLUG="${TASK_SLUG}-${TRAIN_MODE}-${CLASS_WEIGHTING}"; TASK_SLUG="$(sanitize_component "${TASK_SLUG}")"; TASK_CHECKPOINT="$(sanitize_component "${CHECKPOINT_LABEL}")"
TASK_LOG_DIR="${LOG_ROOT}/${LOG_DATASET}/${LOG_ANIMAL}/job-${ARRAY_JOB_ID}"; TASK_STEM="task-$(printf '%03d' "${TASK_INDEX}")__${TASK_SLUG}__${TASK_CHECKPOINT}__k${CANDIDATE_K}"
TASK_OUT_PATH="${TASK_LOG_DIR}/${TASK_STEM}.out"; TASK_ERR_PATH="${TASK_LOG_DIR}/${TASK_STEM}.err"; TASK_COMBINED_PATH="${TASK_LOG_DIR}/${TASK_STEM}.combined.log"; TASK_METADATA_PATH="${TASK_LOG_DIR}/${TASK_STEM}.json"; mkdir -p "${TASK_LOG_DIR}"
exec > >(tee -a "${TASK_OUT_PATH}" "${TASK_COMBINED_PATH}") 2> >(tee -a "${TASK_ERR_PATH}" "${TASK_COMBINED_PATH}" >&2)
COMMAND_STRING="$(printf '%q ' python train/probe.py "${PROBE_ARGS[@]}")"; STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python scripts/probe_log_metadata.py init --path "${TASK_METADATA_PATH}" --job-id "${ARRAY_JOB_ID}" --task-id "${TASK_INDEX}" --dataset "${LOG_DATASET}" --animal "${LOG_ANIMAL}" --method "${METHOD}" --matcher "${MATCHER}" --train-mode "${TRAIN_MODE}" --class-weighting "${CLASS_WEIGHTING}" --checkpoint "${CHECKPOINT_LABEL}" --checkpoint-path "${CHECKPOINT_PATH}" --candidate-k "${CANDIDATE_K}" --command "${COMMAND_STRING}" --start-time "${STARTED_AT}" --stdout-path "${TASK_OUT_PATH}" --stderr-path "${TASK_ERR_PATH}" --combined-path "${TASK_COMBINED_PATH}" --submission-id "${SUBMISSION_ID}" --manifest-path "${PROBE_PARALLEL_MANIFEST:-}" --profile-id "${PROFILE_ID}" --checkpoint-owner "${CHECKPOINT_OWNER:-}" --checkpoint-sha256 "${CHECKPOINT_SHA256:-}" --validation-status validated --status running

finalize_task() {
    local exit_code="$?" status=completed run_directory=""; [[ "${exit_code}" -eq 0 ]] || status=failed
    if [[ -s "${TASK_OUT_PATH}" ]]; then local result_path; result_path="$(sed -n 's/^Saved JSON: //p' "${TASK_OUT_PATH}" | tail -n 1)"; [[ -n "${result_path}" && -f "${result_path}" ]] && run_directory="$(dirname "${result_path}")"; fi
    python scripts/probe_log_metadata.py update --path "${TASK_METADATA_PATH}" --status "${status}" --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --experiment-run-directory "${run_directory}" --error-file "${TASK_ERR_PATH}" --validation-status validated || true
    python scripts/summarize_logs.py --logs-root "${LOG_ROOT}" --write-index --quiet || true
    exit "${exit_code}"
}
trap finalize_task EXIT
nvidia-smi -L
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate ex-reid
python train/probe.py "${PROBE_ARGS[@]}"
