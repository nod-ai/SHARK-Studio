#!/bin/bash
# Sets up a venv suitable for running samples.
# e.g:
# ./setup_venv.sh  #setup a default $PYTHON3 shark.venv
# Environment Variables by the script.
# PYTHON=$PYTHON3.9 ./setup_venv.sh  #pass a version of $PYTHON to use
# VENV_DIR=myshark.venv #create a venv called myshark.venv
# USE_IREE=1 #use stock IREE instead of Nod.ai's SHARK build
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

Red=`tput setaf 1`
Green=`tput setaf 2`
Yellow=`tput setaf 3`

# Assume no binary torch-mlir.
# Currently available for macOS m1&intel (3.9) and Linux(3.7,3.8,3.9,3.10)
torch_mlir_bin=false
if [[ $(uname -s) = 'Darwin' ]]; then
  echo "${Yellow}Apple macOS detected"
  if [[ $(uname -m) == 'arm64' ]]; then
    echo "${Yellow}Apple M1 Detected"
    hash rustc 2>/dev/null
    if [ $? -eq 0 ];then
      echo "${Green}rustc found to compile HF tokenizers"
    else
      echo "${Red}Could not find rustc" >&2
      echo "${Red}Please run:"
      echo "${Red}curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
      exit 1
    fi
  fi
  echo "${Yellow}Run the following commands to setup your SSL certs for your Python version if you see SSL errors with tests"
  echo "${Yellow}/Applications/Python\ 3.XX/Install\ Certificates.command"
  if [ "$PYTHON_VERSION_X_Y" == "3.9" ]; then
    torch_mlir_bin=true
  fi
elif [[ $(uname -s) = 'Linux' ]]; then
  echo "${Yellow}Linux detected"
  if [ "$PYTHON_VERSION_X_Y" == "3.7" ] || [ "$PYTHON_VERSION_X_Y" == "3.8" ]  || [ "$PYTHON_VERSION_X_Y" == "3.9" ] || [ "$PYTHON_VERSION_X_Y" == "3.10" ] ; then
    torch_mlir_bin=true
  fi
else
  echo "${Red}OS not detected. Pray and Play"
fi

# Upgrade pip and install requirements.
$PYTHON -m pip install --upgrade pip || die "Could not upgrade pip"
$PYTHON -m pip install --upgrade -r "$TD/requirements.txt" --extra-index-url https://download.pytorch.org/whl/nightly/cpu -f https://github.com/llvm/torch-mlir/releases
if [ "$torch_mlir_bin" = true ]; then
  $PYTHON -m pip install --find-links https://github.com/llvm/torch-mlir/releases torch-mlir
  if [ $? -eq 0 ];then
    echo "Successfully Installed torch-mlir"
  else
    echo "Could not install torch-mlir" >&2
  fi
else
  echo "${Red}No binaries found for Python $PYTHON_VERSION_X_Y on $(uname -s)"
  echo "${Yello}Python 3.9 supported on macOS and 3.7,3.8,3.9 and 3.10 on Linux"
  echo "${Red}Please build torch-mlir from source in your environment"
  exit 1
fi
if [[ -z "${USE_IREE}" ]]; then
  echo "Installing SHARK..."
  $PYTHON -m pip install --find-links https://github.com/nod-ai/SHARK-Runtime/releases iree-compiler iree-runtime iree-tools-tf iree-tools-tflite
  if [ $? -eq 0 ];then
    echo "Successfully Installed SHARK Runtime"
  else
    echo "Could not install SHARK" >&2
    exit 1
  fi
else
  echo "Installing IREE..."
  $PYTHON -m pip install --find-links https://github.com/google/iree/releases iree-compiler iree-runtime iree-tools-tf iree-tools-tflite
  if [ $? -eq 0 ];then
    echo "Successfully Installed IREE Runtime"
  else
    echo "Could not install IREE" >&2
    exit 1
  fi
fi

$PYTHON -m pip install transformers
$PYTHON -m pip wheel -v -w $TD/wheelhouse $TD -f https://github.com/nod-ai/SHARK-Runtime/releases -f https://github.com/llvm/torch-mlir/releases --extra-index-url https://download.pytorch.org/whl/nightly/cpu
$PYTHON -m pip install . --extra-index-url https://download.pytorch.org/whl/nightly/cpu -f https://github.com/llvm/torch-mlir/releases -f https://github.com/nod-ai/SHARK-Runtime/releases
$PYTHON -m pip install tensorflow

if [[ -z "${CONDA_PREFIX}" ]]; then
  echo "${Green}Before running examples activate venv with:"
  echo "  ${Green}source $VENV_DIR/bin/activate"
fi

