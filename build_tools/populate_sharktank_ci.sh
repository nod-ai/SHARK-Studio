#!/bin/bash

IMPORTER=1 ./setup_venv.sh
source $GITHUB_WORKSPACE/shark.venv/bin/activate
python build_tools/stable_diffusion_testing.py --gen
