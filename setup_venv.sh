#!/bin/bash
# Sets up a venv suitable for running samples.
# e.g:
# ./setup_venv.sh  #setup a default $PYTHON3 shark.venv
# Environment Variables by the script.
# PYTHON=$PYTHON3.10 ./setup_venv.sh  #pass a version of $PYTHON to use
# VENV_DIR=myshark.venv #create a venv called myshark.venv
# USE_IREE=1 #use stock IREE instead of Nod.ai's SHARK build
# IMPORTER=1 #Install importer deps
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
# Currently available for macOS m1&intel (3.10) and Linux(3.7,3.8,3.9,3.10)
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
  if [ "$PYTHON_VERSION_X_Y" == "3.10" ]; then
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

TM_HTML_URL="$(python3 -c "import urllib.request, json, sys; \
	u=json.loads(urllib.request.urlopen('https://api.github.com/repos/llvm/torch-mlir/releases/latest').read().decode()).get('html_url', False);\
	print(u) if u else sys.exit(1);")"
TM_RELEASE_DIR=${TM_HTML_URL/"tag"/"expanded_assets"}
echo "TM_HTML_URL=${TM_HTML_URL}"
echo "TM_RELEASE_DIR=${TM_RELEASE_DIR}"

# Upgrade pip and install requirements.
$PYTHON -m pip install --upgrade pip || die "Could not upgrade pip"
$PYTHON -m pip install --upgrade -r "$TD/requirements.txt"
if [ "$torch_mlir_bin" = true ]; then
  $PYTHON -m pip install --find-links ${TM_RELEASE_DIR} torch-mlir --extra-index-url ${TM_RELEASE_DIR}
  if [ $? -eq 0 ];then
    echo "Successfully Installed torch-mlir"
  else
    echo "Could not install torch-mlir" >&2
  fi
else
  echo "${Red}No binaries found for Python $PYTHON_VERSION_X_Y on $(uname -s)"
  echo "${Yello}Python 3.10 supported on macOS and 3.7,3.8,3.9 and 3.10 on Linux"
  echo "${Red}Please build torch-mlir from source in your environment"
  exit 1
fi
if [[ -z "${USE_IREE}" ]]; then
  RUNTIME="nod-ai/SHARK-Runtime"
else
  RUNTIME="google/iree"
fi
if [[ -z "${NO_BACKEND}" ]]; then
  echo "Installing ${RUNTIME}..."
  $PYTHON -m pip install --find-links https://github.com/${RUNTIME}/releases iree-compiler iree-runtime
else
  echo "Not installing a backend, please make sure to add your backend to PYTHONPATH"
fi
if [[ ! -z "${IMPORTER}" ]]; then
  echo "${Yellow}Installing importer tools.."
  if [[ $(uname -s) = 'Linux' ]]; then
    echo "${Yellow}Linux detected.. installing Linux importer tools"
    $PYTHON -m pip install --upgrade -r "$TD/requirements-importer.txt" -f https://github.com/${RUNTIME}/releases --extra-index-url https://test.pypi.org/simple/ --extra-index-url https://download.pytorch.org/whl/nightly/cu116
  elif [[ $(uname -s) = 'Darwin' ]]; then
    echo "${Yellow}macOS detected.. installing macOS importer tools"
    #Conda seems to have some problems installing these packages and hope they get resolved upstream.
    $PYTHON -m pip install --upgrade -r "$TD/requirements-importer-macos.txt" -f https://github.com/${RUNTIME}/releases --extra-index-url https://download.pytorch.org/whl/nightly/cpu
  fi
fi

$PYTHON -m pip install -e . -f ${${TM_RELEASE_DIR}} -f https://github.com/${RUNTIME}/releases

if [[ $(uname -s) = 'Linux' && ! -z "${IMPORTER}" ]]; then
  $PYTHON -m pip uninstall -y torch torchvision
  $PYTHON -m pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu116
  if [ $? -eq 0 ];then
    echo "Successfully Installed torch + cu116."
  else
    echo "Could not install torch + cu116." >&2
  fi
fi

if [[ ! -z "${ONNX}" ]]; then
  echo "${Yellow}Installing ONNX and onnxruntime for benchmarks..."
  $PYTHON -m pip install onnx onnxruntime psutil
  if [ $? -eq 0 ];then
    echo "Successfully installed ONNX and ONNX runtime."
  else
    echo "Could not install ONNX." >&2
  fi
fi

if [[ -z "${CONDA_PREFIX}" ]]; then
  echo "${Green}Before running examples activate venv with:"
  echo "  ${Green}source $VENV_DIR/bin/activate"
fi

