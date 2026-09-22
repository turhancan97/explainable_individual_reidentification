#!/bin/bash
#SBATCH -p rtx4090_batch
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --exclude=c11,c15,c22
#SBATCH --job-name=probe_reid
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

nvidia-smi -L

conda init bash
source "${EXREID_ROOT:-$PWD}/env.sh"
activate_conda_env "${CONDA_ENV_EXREID}"

python train/probe.py \
        benchmark.method=vismatch \
        benchmark.methods.vismatch.matcher=loma \
        benchmark.methods.vismatch.checkpoint_source=custom \
        benchmark.methods.vismatch.checkpoint_path=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/loma-b-finetuned-trainval-4gpu/epoch_299/model.safetensors \
        benchmark.methods.vismatch.checkpoint_components=matcher_only