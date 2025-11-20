#!/bin/bash

TD="$(cd $(dirname $0) && pwd)"
if [ -z "$PYTHON" ]; then
  PYTHON="$(which python3)"
fi

function die() {
  echo "Error executing command: $*"
  exit 1
}

PYTHON_VERSION_X_Y=`${PYTHON} -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))'`

echo "Python: $PYTHON"
echo "Python version: $PYTHON_VERSION_X_Y"

mkdir input_images

wget https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg -P input_images/
wget https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg -P input_images/
wget https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg -P input_images/
wget https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg -P input_images/

pip install diffusers["training"]==0.4.1 transformers ftfy opencv-python
