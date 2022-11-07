#!/bin/bash

source /etc/profile
module load anaconda/2022a
module load cuda/11.3

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

export PYTHONPATH=.
export TFHUB_CACHE_DIR="$(pwd)/models/cache"
python experiments/training/plant_classifiers_final.py $LLSUB_RANK $LLSUB_SIZE

# Might need to run chmod u+x job_array.sh
