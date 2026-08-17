#!/bin/bash
#SBATCH -p rtx4090_batch
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --exclude=c22
#SBATCH --job-name=probe_reid
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

nvidia-smi -L

conda init bash
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate ex-reid

python train/probe.py \
        benchmark.method=vismatch \
        benchmark.methods.vismatch.matcher=loma \
        benchmark.methods.vismatch.checkpoint_source=custom \
        benchmark.methods.vismatch.checkpoint_path=/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/loma-b-wandb/epoch_110/model.safetensors \
        benchmark.methods.vismatch.checkpoint_components=matcher_only