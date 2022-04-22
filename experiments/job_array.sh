#!/bin/bash

source /etc/profile
module load anaconda/2022a
module load cuda/11.6

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

TFHUB_CACHE_DIR="$(pwd)/models/cache" python experiments/scripts/parameter_text_models.py $LLSUB_RANK $LLSUB_SIZE

# Run this file via LLsub ./experiments/job_array.sh -s cores -g volta:1 [nodes,processes_per_node,threads_per_process]
# Number of cores controls the RAM - each core gives 8GB of RAM
# Might need to run chmod u+x job_array.sh
