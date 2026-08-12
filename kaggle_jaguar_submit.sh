#!/bin/bash
#SBATCH -p dgxa100
#SBATCH --gpus=1
#SBATCH --qos=big
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=kaggle_jaguar_submit
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

nvidia-smi -L

conda init bash
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate ex-reid

python scripts/kaggle_jaguar_submit.py --config config/kaggle_jaguar.yaml