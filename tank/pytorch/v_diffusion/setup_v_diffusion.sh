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

git clone --recursive https://github.com/crowsonkb/v-diffusion-pytorch.git

pip install -r v-diffusion-pytorch/requirements.txt
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git


mkdir v-diffusion-pytorch/checkpoints
wget https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth -P v-diffusion-pytorch/checkpoints/
