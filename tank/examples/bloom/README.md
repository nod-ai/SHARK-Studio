# Bloom model

## Installation

<details>
  <summary>Installation (Linux)</summary>

### Activate shark.venv Virtual Environment

```shell
source shark.venv/bin/activate

# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
```

### Install dependencies

```shell
pip install transformers==4.21.2
```
Use this branch of Torch-MLIR for running the model: https://github.com/vivekkhandelwal1/torch-mlir/tree/bloom-ops


### Run bloom model

```shell
python bloom_model.py
```

The runtime device, model config, and text prompt can be specified with `--device <device string>`, `--config <config string>`, `--prompt <prompt string>` respectively.

To run the complete 176B params bloom model, run the following command:
```shell
python bloom_model.py --config "bloom"
```
