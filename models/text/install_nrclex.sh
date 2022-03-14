#!/bin/bash

cd $(dirname "$BASH_SOURCE")

cd ../..


if [ -f "venv/bin/python" ]; then
    venv/bin/python -m textblob.download_corpora
else
    # In CI or docker there is no venv
    export PYTHONPATH=.
    python -m textblob.download_corpora
fi
