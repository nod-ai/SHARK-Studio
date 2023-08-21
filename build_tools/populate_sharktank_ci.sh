#!/bin/bash

IMPORTER=1 BENCHMARK=1 NO_BREVITAS=1 ./setup_venv.sh
source $GITHUB_WORKSPACE/shark.venv/bin/activate
python build_tools/stable_diffusion_testing.py --gen
python tank/generate_sharktank.py
