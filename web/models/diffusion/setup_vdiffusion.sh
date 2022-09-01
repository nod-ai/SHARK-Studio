git clone --recursive https://github.com/crowsonkb/v-diffusion-pytorch.git
pip install ftfy regex tqdm

mkdir checkpoints
wget https://the-eye.eu/public/AI/models/v-diffusion/cc12m_1_cfg.pth -P checkpoints/
