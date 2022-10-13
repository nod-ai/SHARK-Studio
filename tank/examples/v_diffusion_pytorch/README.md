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

The runtime device can be specified with `--runtime_device=<device string>`

### Run the v-diffusion model via torch-mlir
```shell
./cfg_sample.py "New York City, oil on canvas":5 -n 1 -bs 1 --steps 2
```

### Run the model stored in the tank
```shell
./cfg_sample_from_mlir.py "New York City, oil on canvas":5 -n 1 -bs 1 --steps 2
```
Note that the current model in the tank requires batch size 1 statically.

### Run the model with preprocessing elements taken out
To run the model without preprocessing copy `cc12m_1.py` to replace the version in `v-diffusion-pytorch`
```shell
cp cc12m_1.py v-diffusion-pytorch/diffusion/models
```
Then run
```shell
./cfg_sample_preprocess.py "New York City, oil on canvas":5 -n 1 -bs 1 --steps 2
```
