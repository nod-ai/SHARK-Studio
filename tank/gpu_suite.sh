#! /bin/sh
pytest tank/ -k "gpu" --ignore=tank/tf/
pytest tank/tf/ -k "gpu"
