#!/bin/bash

cd $(dirname "$BASH_SOURCE")

git clone https://github.com/GasserElbanna/serab-byols.git
python -m pip install -e ./serab-byols
