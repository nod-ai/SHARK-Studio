# SHARK

High Performance Machine Learning and Data Analytics for CPUs, GPUs, Accelerators and Heterogeneous Clusters

[![Nightly Release](https://github.com/nod-ai/SHARK/actions/workflows/nightly.yml/badge.svg)](https://github.com/nod-ai/SHARK/actions/workflows/nightly.yml)

## Communication Channels

*   [Nod.ai SHARK Discord server](https://discord.gg/RUqY2h2s9u): Real time discussions with the nod.ai team and other users
*   [GitHub issues](https://github.com/nod-ai/SHARK/issues): Feature requests, bugs etc


## Installation

<details>
  <summary>Installation (Linux and macOS)</summary>
  
### Setup a new pip Virtual Environment

This step sets up a new VirtualEnv for Python
  
```shell
python --version #Check you have 3.7->3.10 on Linux or 3.10 on macOS
python -m venv shark_venv
source shark_venv/bin/activate

# If you are using conda create and activate a new conda env

# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
```

*macOS Metal* users please install https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.dmg

### Install SHARK
  
This step pip installs SHARK and related packages on Linux Python 3.7, 3.8, 3.9, 3.10 and macOS Python 3.10

```shell
pip install nodai-shark -f https://github.com/nod-ai/SHARK/releases -f https://github.com/llvm/torch-mlir/releases -f https://github.com/nod-ai/shark-runtime/releases --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```
If you are on an Intel macOS machine you need this [workaround](https://github.com/nod-ai/SHARK/issues/102) for an upstream issue.

### Download and run Resnet50 sample
    
```shell
curl -O https://raw.githubusercontent.com/nod-ai/SHARK/main/shark/examples/shark_inference/resnet50_script.py
#Install deps for test script
pip install --pre torch torchvision torchaudio tqdm pillow --extra-index-url https://download.pytorch.org/whl/nightly/cpu
python ./resnet50_script.py --device="cpu"  #use cuda or vulkan or metal 
```
        
### Download and run BERT (MiniLM) sample
```shell
curl -O https://raw.githubusercontent.com/nod-ai/SHARK/main/shark/examples/shark_inference/minilm_jit.py
#Install deps for test script
pip install transformers torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
python ./minilm_jit.py --device="cpu"  #use cuda or vulkan or metal 
```
</details>


<details>
  <summary>Source Installation</summary>

## Check out the code

```shell
git clone https://github.com/nod-ai/SHARK.git 
```

## Setup your Python VirtualEnvironment and Dependencies
```shell
# Setup venv and install necessary packages (torch-mlir, nodLabs/Shark, ...).
./setup_venv.sh
# Please activate the venv after installation.
```

### Run a demo script
```shell
python -m  shark.examples.shark_inference.resnet50_script --device="cpu" # Use gpu | vulkan
```


### Run all tests on CPU/GPU/VULKAN/Metal
```shell
pytest

# If on Linux for quicker results:
pytest --workers auto
```
</details>


<details>
  <summary>API Reference</summary>

### Shark Inference API

```
from shark_runner import SharkInference

shark_module = SharkInference(
        module = model class.
        (input,)  = inputs to model (must be a torch-tensor)
        dynamic (boolean) = Pass the input shapes as static or dynamic.
        device = `cpu`, `gpu` or `vulkan` is supported.
        tracing_required = (boolean) = Jit trace the module with the given input, useful in the case where jit.script doesn't work. )
shark_module.set_frontend("pytorch") # Use tensorflow, mhlo, linalg, tosa
shark_module.compile()

result = shark_module.forward(inputs)
```


### Example demonstrating running MHLO IR.

```
from shark.shark_inference import SharkInference
import numpy as np

mhlo_ir = r"""builtin.module  {
      func.func @forward(%arg0: tensor<1x4xf32>, %arg1: tensor<4x1xf32>) -> tensor<4x4xf32> {
        %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<1x4xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
        %1 = "mhlo.abs"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
        return %1 : tensor<4x4xf32>
      }
}"""

arg0 = np.ones((1, 4)).astype(np.float32)
arg1 = np.ones((4, 1)).astype(np.float32)

shark_module = SharkInference(mhlo_ir, (arg0, arg1))
shark_module.set_frontend("mhlo")
shark_module.compile()
print(shark_module.forward((arg0, arg1)))
```
</details>


## Supported and Validated Models

<details>
  <summary>PyTorch Models</summary>

### Huggingface PyTorch Models

| Hugging Face Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :heavy_check_mark: (JIT)          | :heavy_check_mark:         |          |             |
| Albert              | :heavy_check_mark: (JIT)            | :heavy_check_mark:         |          |             |
| BigBird             | :heavy_check_mark: (AOT)            |          |          |             |
| DistilBERT          | :heavy_check_mark: (JIT)            | :heavy_check_mark:         |          |             |
| GPT2                | :x: (AOT)            |          |          |             |

### Torchvision  Models
  
| TORCHVISION Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|--------------------|----------------------|----------|----------|-------------|
| AlexNet            | :heavy_check_mark: (Script)         | :heavy_check_mark:         | :heavy_check_mark:         |             |
| DenseNet121        | :heavy_check_mark: (Script)         |          |          |             |
| MNasNet1_0         | :heavy_check_mark: (Script)         |          |          |             |
| MobileNetV2        | :heavy_check_mark: (Script)         |          |          |             |
| MobileNetV3        | :heavy_check_mark: (Script)         |          |          |             |
| Unet               | :x: (Script)         |          |          |             |
| Resnet18           | :heavy_check_mark: (Script)         | :heavy_check_mark:         |  :heavy_check_mark:        |             |
| Resnet50           | :heavy_check_mark: (Script)         | :heavy_check_mark:         |   :heavy_check_mark:       |             |
| Resnet101           | :heavy_check_mark: (Script)         | :heavy_check_mark:         |   :heavy_check_mark:       |             |
| Resnext50_32x4d    | :heavy_check_mark: (Script)         |          |          |             |
| ShuffleNet_v2      | :x: (Script)         |          |          |             |
| SqueezeNet         | :heavy_check_mark: (Script)         | :heavy_check_mark:         |   :heavy_check_mark:       |             |
| EfficientNet       | :heavy_check_mark: (Script)         |          |          |             |
| Regnet             | :heavy_check_mark: (Script)         |          |          |             |
| Resnest            | :x: (Script)         |          |          |             |
| Vision Transformer | :heavy_check_mark: (Script)         |          |          |             |
| VGG 16             | :heavy_check_mark: (Script)         | :heavy_check_mark:         |   :heavy_check_mark:       |             |
| Wide Resnet        | :heavy_check_mark: (Script)         | :heavy_check_mark:         | :heavy_check_mark:         |             |
| RAFT               | :x: (JIT)            |          |          |             |

For more information refer to [MODEL TRACKING SHEET](https://docs.google.com/spreadsheets/d/15PcjKeHZIrB5LfDyuw7DGEEE8XnQEX2aX8lm8qbxV8A/edit#gid=0)

### PyTorch Training Models 

| Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :x:           | :x:         |          |             |
| FullyConnected                | :heavy_check_mark:           | :heavy_check_mark:         |          |             |

</details>
  
<details>
  <summary>JAX Models</summary>


### JAX  Models 

| Models | JAX-MHLO lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| DALL-E                | :x:           | :x:         |          |             |
| FullyConnected                | :heavy_check_mark:           | :heavy_check_mark:         |          |             |
 
</details>
  
<details>
  <summary>TFLite Models</summary>
 
### TFLite Models 

| Models | TOSA/LinAlg  | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :x:           | :x:         |          |             |
| FullyConnected                | :heavy_check_mark:           | :heavy_check_mark:         |          |             |
  
</details>

<details>
  <summary>TF Models</summary>
 
### Tensorflow Models 

| Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :x:           | :x:         |          |             |
| FullyConnected                | :heavy_check_mark:           | :heavy_check_mark:         |          |             |
  
</details>

## Related Projects
  
<details>
  <summary>IREE Project Channels</summary>

*   [Upstream IREE issues](https://github.com/google/iree/issues): Feature requests,
    bugs, and other work tracking
*   [Upstream IREE Discord server](https://discord.gg/26P4xW4): Daily development
    discussions with the core team and collaborators
*   [iree-discuss email list](https://groups.google.com/forum/#!forum/iree-discuss):
    Announcements, general and low-priority discussion
</details>
    
<details>
  <summary>MLIR and Torch-MLIR Project Channels</summary>

* `#torch-mlir` channel on the LLVM [Discord](https://discord.gg/xS7Z362) - this is the most active communication channel
* Torch-MLIR Github issues [here](https://github.com/llvm/torch-mlir/issues)
* [`torch-mlir` section](https://llvm.discourse.group/c/projects-that-want-to-become-official-llvm-projects/torch-mlir/41) of LLVM Discourse
*  Weekly meetings on Mondays 9AM PST. See [here](https://discourse.llvm.org/t/community-meeting-developer-hour-refactoring-recurring-meetings/62575) for more information.
* [MLIR topic within LLVM Discourse](https://llvm.discourse.group/c/llvm-project/mlir/31) SHARK and IREE is enabled by and heavily relies on [MLIR](https://mlir.llvm.org).
</details>
  
## License

nod.ai SHARK is licensed under the terms of the Apache 2.0 License with LLVM Exceptions.
See [LICENSE](LICENSE) for more information.
