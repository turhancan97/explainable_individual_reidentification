#!/bin/bash
#SBATCH -p rtx4090_batch
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=c11,c15,c22
#SBATCH --job-name=finetune_reid
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

nvidia-smi -L

conda init bash
source "${EXREID_ROOT:-$PWD}/env.sh"
activate_conda_env "${CONDA_ENV_EXREID}"

python train/finetune.py