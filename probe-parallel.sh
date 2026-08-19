#!/bin/bash -l
#SBATCH -p rtx4090_batch
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --exclude=c22,c11,c15,dgx1
#SBATCH --job-name=probe_parallel
#SBATCH --time=24:00:00
#SBATCH --output=logs/parallel_run/%x_%A_%a.out
#SBATCH --error=logs/parallel_run/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

# Edit these values to change the experiment grid or Slurm throttle.
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-12}"
CANDIDATE_K_VALUES=(10 50 100 250 500 1000)

LOMA_CUSTOM_CHECKPOINT_PATH="${LOMA_CUSTOM_CHECKPOINT_PATH:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/loma-b-finetuned-trainval-4gpu/epoch_299/model.safetensors}"
RDD_CUSTOM_CHECKPOINT_PATH="${RDD_CUSTOM_CHECKPOINT_PATH:-/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/matches-lg-wandb/epoch_299/model.safetensors}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs/parallel_run

# Format: method|matcher|checkpoint_label|checkpoint_path
# Keep this table explicit so the scientific comparison grid is auditable.
VARIANTS=(
    "cosine|-|default|-"
    "wildfusion|-|default|-"
    # "local_lightglue|-|default|-"
    # "linear_probe|-|default|-"
    # "efficient_probe|-|default|-"
    "vismatch|loma|default|-"
    "vismatch|loma|custom|${LOMA_CUSTOM_CHECKPOINT_PATH}"
    "vismatch|rdd-lightglue|default|-"
    "vismatch|rdd-lightglue|custom|${RDD_CUSTOM_CHECKPOINT_PATH}"
)

TASKS=()
for candidate_k in "${CANDIDATE_K_VALUES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        TASKS+=("${candidate_k}|${variant}")
    done
done

die() {
    echo "probe-parallel.sh: $*" >&2
    exit 1
}

validate_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer; got '${value}'"
}

validate_nonnegative_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer; got '${value}'"
}

validate_custom_checkpoint_paths() {
    [[ -e "${LOMA_CUSTOM_CHECKPOINT_PATH}" ]] || die "LoMa custom checkpoint does not exist: ${LOMA_CUSTOM_CHECKPOINT_PATH}"
    [[ -e "${RDD_CUSTOM_CHECKPOINT_PATH}" ]] || die "RDD-LightGlue custom checkpoint does not exist: ${RDD_CUSTOM_CHECKPOINT_PATH}"
}

print_task() {
    local index="$1"
    local task="$2"
    local candidate_k variant method matcher checkpoint_label checkpoint_path
    IFS='|' read -r candidate_k variant <<< "${task}"
    IFS='|' read -r method matcher checkpoint_label checkpoint_path <<< "${variant}"
    printf 'index=%s candidate_k=%s method=%s matcher=%s checkpoint=%s path=%s\n' \
        "${index}" "${candidate_k}" "${method}" "${matcher}" \
        "${checkpoint_label}" "${checkpoint_path}"
}

build_probe_args() {
    local candidate_k="$1"
    local variant="$2"
    local method matcher checkpoint_label checkpoint_path
    IFS='|' read -r method matcher checkpoint_label checkpoint_path <<< "${variant}"

    PROBE_ARGS=("benchmark.method=${method}" "benchmark.candidate_k=${candidate_k}")
    if [[ "${method}" == "vismatch" ]]; then
        PROBE_ARGS+=("benchmark.methods.vismatch.matcher=${matcher}")
        if [[ "${checkpoint_label}" == "custom" ]]; then
            PROBE_ARGS+=(
                "benchmark.methods.vismatch.checkpoint_source=custom"
                "benchmark.methods.vismatch.checkpoint_path=${checkpoint_path}"
                "benchmark.methods.vismatch.checkpoint_components=matcher_only"
            )
        else
            PROBE_ARGS+=("benchmark.methods.vismatch.checkpoint_source=default")
        fi
    fi
}

print_probe_command() {
    printf 'python train/probe.py'
    printf ' %q' "${PROBE_ARGS[@]}"
    printf '\n'
}

validate_positive_integer "MAX_CONCURRENT_JOBS" "${MAX_CONCURRENT_JOBS}"
for candidate_k in "${CANDIDATE_K_VALUES[@]}"; do
    validate_positive_integer "candidate_k" "${candidate_k}"
done

if [[ "${1:-}" == "--list-tasks" ]]; then
    for index in "${!TASKS[@]}"; do
        print_task "${index}" "${TASKS[${index}]}"
    done
    exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    validate_custom_checkpoint_paths
    LAST_TASK_INDEX=$((${#TASKS[@]} - 1))
    ARRAY_SPEC="0-${LAST_TASK_INDEX}%${MAX_CONCURRENT_JOBS}"
    echo "Submitting ${#TASKS[@]} probe tasks with array throttle ${MAX_CONCURRENT_JOBS}."
    if [[ "${PROBE_PARALLEL_DRY_RUN:-0}" == "1" || "${1:-}" == "--dry-run" ]]; then
        printf 'sbatch --array=%q %q\n' "${ARRAY_SPEC}" "${BASH_SOURCE[0]}"
        exit 0
    fi
    exec sbatch --array="${ARRAY_SPEC}" "${BASH_SOURCE[0]}"
fi

validate_custom_checkpoint_paths
TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
validate_nonnegative_integer "SLURM_ARRAY_TASK_ID" "${TASK_INDEX}"
(( TASK_INDEX < ${#TASKS[@]} )) || die "array task index ${TASK_INDEX} is outside 0..$((${#TASKS[@]} - 1))"

CURRENT_TASK="${TASKS[${TASK_INDEX}]}"
IFS='|' read -r CANDIDATE_K VARIANT <<< "${CURRENT_TASK}"
IFS='|' read -r METHOD MATCHER CHECKPOINT_LABEL CHECKPOINT_PATH <<< "${VARIANT}"
build_probe_args "${CANDIDATE_K}" "${VARIANT}"

echo "Starting probe array task ${TASK_INDEX}/${#TASKS[@]}"
print_task "${TASK_INDEX}" "${CURRENT_TASK}"
print_probe_command

if [[ "${PROBE_PARALLEL_DRY_RUN:-0}" == "1" || "${1:-}" == "--dry-run" ]]; then
    exit 0
fi

nvidia-smi -L
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate ex-reid
python train/probe.py "${PROBE_ARGS[@]}"
