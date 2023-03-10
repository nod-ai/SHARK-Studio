#!/bin/bash

IMPORTER=1 BENCHMARK=1 ./setup_venv.sh
source $GITHUB_WORKSPACE/shark.venv/bin/activate
python tank/generate_sharktank.py
