#!/bin/bash
#SBATCH --job-name=infer_black_man
#SBATCH --output=logs/infer_black_man.out
#SBATCH --error=logs/infer_black_man.err
#SBATCH --partition=gpu         # or gpu_test if testing
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH --cpus-per-task=4       # CPU cores
#SBATCH --mem=32G               # RAM
#SBATCH --time=09:00:00         # Max runtime (HH:MM:SS)

export PYTHONPATH="$PWD/third_party/segment-anything:$PYTHONPATH"
module load cuda/11.8.0-fasrc01
module load gcc/10.2.0-fasrc01
module load Mambaforge/23.3.1-fasrc01
conda activate humangaussian

python launch.py --config configs/test.yaml --train --gpu 0 system.prompt_processor.prompt="A black man wearing a red baseball cap" system.masking=True system.masking_each_own=True