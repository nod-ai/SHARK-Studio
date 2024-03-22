#!/bin/bash
# Sets up a venv suitable for running samples.
# e.g:
# ./setup_venv.sh  #setup a default $PYTHON3 shark.venv
# Environment variables used by the script.
# PYTHON=$PYTHON3.10 ./setup_venv.sh  #pass a version of $PYTHON to use
# VENV_DIR=myshark.venv #create a venv called myshark.venv
# SKIP_VENV=1 #Don't create and activate a Python venv. Use the current environment. 
# USE_IREE=1 #use stock IREE instead of Nod.ai's SHARK build
# IMPORTER=1 #Install importer deps
# BENCHMARK=1 #Install benchmark deps
# NO_BACKEND=1 #Don't install iree or shark backend
# if you run the script from a conda env it will install in your conda env

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

if [ "$PYTHON_VERSION_X_Y" != "3.11" ]; then
    echo "Error: Python version 3.11 is required."
    exit 1
fi

if [[ "$SKIP_VENV" != "1" ]]; then
  if [[ -z "${CONDA_PREFIX}" ]]; then
    # Not a conda env. So create a new VENV dir
    VENV_DIR=${VENV_DIR:-shark.venv}
    echo "Using pip venv.. Setting up venv dir: $VENV_DIR"
    $PYTHON -m venv "$VENV_DIR" || die "Could not create venv."
    source "$VENV_DIR/bin/activate" || die "Could not activate venv"
    PYTHON="$(which python3)"
  else
    echo "Found conda env $CONDA_DEFAULT_ENV. Running pip install inside the conda env"
  fi
fi

Red=`tput setaf 1`
Green=`tput setaf 2`
Yellow=`tput setaf 3`

# Upgrade pip and install requirements.
$PYTHON -m pip install --upgrade pip || die "Could not upgrade pip"
$PYTHON -m pip install --upgrade -r "$TD/requirements.txt"
if [[ $(uname -s) = 'Darwin' ]]; then
  echo "MacOS detected. Installing torch-mlir from .whl, to avoid dependency problems with torch."
  $PYTHON -m pip uninstall -y timm #TEMP FIX FOR MAC
  $PYTHON -m pip install --pre --no-cache-dir torch-mlir -f https://llvm.github.io/torch-mlir/package-index/ -f https://download.pytorch.org/whl/nightly/torch/
else
  $PYTHON -m pip install --pre torch-mlir -f https://llvm.github.io/torch-mlir/package-index/
  if [ $? -eq 0 ];then
    echo "Successfully Installed torch-mlir"
  else
    echo "Could not install torch-mlir" >&2
  fi
fi
if [[ -z "${USE_IREE}" ]]; then
  rm .use-iree
  RUNTIME="https://nod-ai.github.io/SRT/pip-release-links.html"
else
  touch ./.use-iree
  RUNTIME="https://openxla.github.io/iree/pip-release-links.html"
fi
if [[ -z "${NO_BACKEND}" ]]; then
  echo "Installing ${RUNTIME}..."
  $PYTHON -m pip install --pre --upgrade --no-index --find-links ${RUNTIME} iree-compiler iree-runtime
else
  echo "Not installing a backend, please make sure to add your backend to PYTHONPATH"
fi

if [[ $(uname -s) = 'Darwin' ]]; then
  PYTORCH_URL=https://download.pytorch.org/whl/nightly/torch/
else
  PYTORCH_URL=https://download.pytorch.org/whl/nightly/cpu/
fi

$PYTHON -m pip install --no-warn-conflicts -e . -f https://llvm.github.io/torch-mlir/package-index/ -f ${RUNTIME} -f ${PYTORCH_URL}

if [[ $(uname -s) = 'Linux' && ! -z "${IMPORTER}" ]]; then
  T_VER=$($PYTHON -m pip show torch | grep Version)
  T_VER_MIN=${T_VER:14:12}
  TV_VER=$($PYTHON -m pip show torchvision | grep Version)
  TV_VER_MAJ=${TV_VER:9:6}
  $PYTHON -m pip uninstall -y torchvision
  $PYTHON -m pip install torchvision==${TV_VER_MAJ}${T_VER_MIN} --no-deps -f https://download.pytorch.org/whl/nightly/cpu/torchvision/
  if [ $? -eq 0 ];then
    echo "Successfully Installed torch + cu118."
  else
    echo "Could not install torch + cu118." >&2
  fi
fi

if [[ -z "${NO_BREVITAS}" ]]; then
  $PYTHON -m pip install git+https://github.com/Xilinx/brevitas.git@dev
fi

if [[ -z "${CONDA_PREFIX}" && "$SKIP_VENV" != "1" ]]; then
  echo "${Green}Before running examples activate venv with:"
  echo "  ${Green}source $VENV_DIR/bin/activate"
fi
