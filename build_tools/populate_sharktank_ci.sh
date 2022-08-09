#!/bin/bash

IMPORTER=1 ./setup_venv.sh
source $GITHUB_WORKSPACE/shark.venv/bin/activate
python generate_sharktank.py --upload=False
