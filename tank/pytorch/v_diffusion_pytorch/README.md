# v-diffusion model

## Installation

<details>
  <summary>Installation (Linux)</summary>

### Activate shark.venv Virtual Environment

```shell
source shark.venv/bin/activate

# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
```

### Install v-diffusion model and its dependencies

```shell
cd tank/pytorch/v_diffusion/
Run the script setup_v_diffusion_pytorch.sh
```

### Run v-diffusion-pytorch model

```shell
./v-diffusion-pytorch/cfg_sample.py "New York City, oil on canvas":5 -n 5 -bs 5
```

### Run the v-diffusion model via torch-mlir
```shell
./cfg_sample.py "New York City, oil on canvas":5 -n 1 -bs 1 --steps 2
```
