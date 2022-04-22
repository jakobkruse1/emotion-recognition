#!/bin/bash

# Slurm sbatch options
#SBATCH -o experiment.log-%j
#SBATCH -n 8
#SBATCH -c 20
#SBATCH -N 4
#SBATCH --gres=gpu:volta:1
#SBATCH --exclusive

source /etc/profile
module load anaconda/2022a
module load cuda/11.6

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

TFHUB_CACHE_DIR="$(pwd)/models/cache" python experiments/scripts/parameter_text_models.py $LLSUB_RANK $LLSUB_SIZE

# Might need to run chmod u+x job_array.sh
