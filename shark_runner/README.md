# Shark Runner

The Shark Runner provides inference and training APIs to run deep learning models on Shark Runtime.

# How to configure.

## Check out the code

```shell
git clone https://github.com/NodLabs/dSHARK.git 
cd dSHARK
```

## Setup your Python VirtualEnvironment and Dependencies
```shell
python -m venv shark_venv
source shark_venv/bin/activate
# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
# Install latest PyTorch nightlies and build requirements.
python -m pip install -r requirements.txt
```

## Install dependent packages
```shell
# Install latest torch-mlir release.
python -m pip install --find-links https://github.com/llvm/torch-mlir/releases torch-mlir

# Install latest IREE release.
python -m pip install --find-links https://github.com/google/iree/releases iree-compiler iree-runtime

# Install functorch
python -m pip install ninja
python -m pip install "git+https://github.com/pytorch/functorch.git"

# Install shark_runner from the current path.
python -m pip install .
```


### Run a demo script
```shell
cd shark_runner/examples/
python resnet50_script.py
```

### Shark Inference API

```
from shark_runner import SharkInference

shark_module = SharkInference(
        module = torch.nn.module class.
        (input,)  = inputs to model (must be a torch-tensor)
        dynamic (boolean) = Pass the input shapes as static or dynamic.
        device = `cpu`, `gpu` or `vulkan` is supported.
        tracing_required = (boolean) = Jit trace the module with the given input, useful in the case where jit.script doesn't work. )

result = shark_module.forward(inputs)
```

### Shark Trainer API

#### Work in Progress


### Model Tracking (Shark Inference)

| Hugging Face Models | Torch-MLIR lowerable | IREE-CPU | IREE-GPU | IREE-VULKAN |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :heavy_check_mark: (JIT)          |          |          |             |
| Albert              | :heavy_check_mark: (JIT)            |          |          |             |
| BigBird             | :heavy_check_mark: (AOT)            |          |          |             |
| DistilBERT          | :heavy_check_mark: (AOT)            |          |          |             |
| GPT2                | :x: (AOT)            |          |          |             |


| TORCHVISION Models | Torch-MLIR lowerable | IREE-CPU | IREE-GPU | IREE-VULKAN |
|--------------------|----------------------|----------|----------|-------------|
| AlexNet            | :heavy_check_mark: (Script)         |          |          |             |
| DenseNet121        | :heavy_check_mark: (Script)         |          |          |             |
| MNasNet1_0         | :heavy_check_mark: (Script)         |          |          |             |
| MobileNetV2        | :heavy_check_mark: (Script)         |          |          |             |
| MobileNetV3        | :heavy_check_mark: (Script)         |          |          |             |
| Unet               | :x: (Script)         |          |          |             |
| Resnet18           | :heavy_check_mark: (Script)         |          |          |             |
| Resnet50           | :heavy_check_mark: (Script)         |          |          |             |
| Resnext50_32x4d    | :heavy_check_mark: (Script)         |          |          |             |
| ShuffleNet_v2      | :x: (Script)         |          |          |             |
| SqueezeNet         | :x: (Script)         |          |          |             |
| EfficientNet       | :heavy_check_mark: (Script)         |          |          |             |
| Regnet             | :heavy_check_mark: (Script)         |          |          |             |
| Resnest            | :x: (Script)         |          |          |             |
| Vision Transformer | :heavy_check_mark: (Script)         |          |          |             |
| VGG 16             | :heavy_check_mark: (Script)         |          |          |             |
| Wide Resnet        | :heavy_check_mark: (Script)         |          |          |             |
| RAFT               | :x: (JIT)            |          |          |             |

For more information refer to [MODEL TRACKING SHEET](https://docs.google.com/spreadsheets/d/15PcjKeHZIrB5LfDyuw7DGEEE8XnQEX2aX8lm8qbxV8A/edit#gid=0)
