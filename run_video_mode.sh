#!/bin/bash
#SBATCH --job-name=humangaussian_video
#SBATCH --output=logs/video_%j.out
#SBATCH --error=logs/video_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00

export PYTHONPATH="$PWD/third_party/segment-anything:$PYTHONPATH"
mkdir -p logs

# Stage 1: Training (GCC 12.2.0)
module load gcc/12.2.0-fasrc01
module load cuda/11.8.0-fasrc01
module load Mambaforge/23.3.1-fasrc01
mamba activate HGFresh

python launch.py --config "${1:-configs/test_video.yaml}" --train --gpu 0

# Stage 2: Animation (GCC 9.5.0)
TRIAL_DIR=$(ls -td outputs/*/ | head -1)
ANIMATION_SCRIPT="${TRIAL_DIR}run_animation.sh"

if [ -f "$ANIMATION_SCRIPT" ]; then
    module purge
    module load gcc/9.5.0-fasrc01
    module load cuda/11.8.0-fasrc01
    python
    mamba activate HGFresh
    bash "$ANIMATION_SCRIPT"
fi
