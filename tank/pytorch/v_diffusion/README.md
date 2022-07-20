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
./setup_diffusion.sh
```

### Run v-diffusion-pytorch model

```shell
./v-diffusion-pytorch/cfg_sample.py "the rise of consciousness":5 -n 5 -bs 5 --seed 0
```

### Compile v-diffusion model via torch-mlir
```shell
python v_diffusion.py 2> v_diffusion_ir.mlir
```
