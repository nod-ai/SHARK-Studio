#!/bin/bash
# Sets up a venv suitable for running samples.
# Recommend getting default 'python' to be python 3. For example on Debian:
#   sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
# Or launch with python=/some/path
TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="$TD/shark.venv"
if [ -z "$PYTHON" ]; then
  PYTHON="$(which python3)"
fi

echo "Setting up venv dir: $VENV_DIR"
echo "Python: $PYTHON"
echo "Python version: $("$PYTHON" --version)"

function die() {
  echo "Error executing command: $*"
  exit 1
}

$PYTHON -m venv "$VENV_DIR" || die "Could not create venv."
source "$VENV_DIR/bin/activate" || die "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || die "Could not upgrade pip"
python -m pip install --upgrade -r "$TD/requirements.txt"
python -m pip install --find-links https://github.com/llvm/torch-mlir/releases torch-mlir
python -m pip install --find-links https://github.com/NodLabs/SHARK/releases iree-compiler iree-runtime
#python -m pip install --find-links https://github.com/google/iree/releases iree-compiler iree-runtime

Red=`tput setaf 1`
Green=`tput setaf 2`
Yellow=`tput setaf 3`

if [[ $(uname -s) = 'Darwin' ]]; then
  if [[ $(uname -m) == 'arm64' ]]; then
    echo "${Yellow}Apple M1 Detected"
    echo "${Yellow}If tokenizers install fails make sure you have the latest rustc"
    echo "${Yellow}Please run:"
    echo "${Yellow}curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  fi
fi

python -m pip install transformers
python -m pip install git+https://github.com/pytorch/functorch.git
pip install . --extra-index-url https://download.pytorch.org/whl/nightly/cpu -f https://github.com/llvm/torch-mlir/releases

echo "${Green}Before running examples activate venv with:"
echo "  ${Green}source $VENV_DIR/bin/activate"

