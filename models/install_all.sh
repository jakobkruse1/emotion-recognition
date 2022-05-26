#!/bin/bash

cd $(dirname "$BASH_SOURCE")

# Setup
DOCKER_FLAG=""
FORCE_FLAG=""
while getopts ":df" opt; do
  case $opt in
    d)
      # If -d flag is specified, run installation without sudo
      echo "Using docker configuration without sudo!"
      DOCKER_FLAG="-d"
      ;;
    f)
      # Force flag that overwrites data also if it is already there
      FORCE_FLAG="-f"
      ;;
    \?)
      echo "Unexpected option -$OPTARG"
      ;;
  esac
done

bash text/install_nrclex.sh $DOCKER_FLAG $FORCE_FLAG

cd ..
TFHUB_CACHE_DIR="$(pwd)/models/cache" python models/cache_models.py
