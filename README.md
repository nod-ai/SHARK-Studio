# SHARK


## Communication Channels

*   [GitHub issues](https://github.com/nod-ai/SHARK/issues): Feature requests, bugs etc
*   [Nod.ai SHARK Discord server](https://discord.gg/RUqY2h2s9u): Real time discussions with the nod.ai team and other users


# Installation

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


### Run all tests on CPU/GPU/VULKAN
```shell
pytest

# If on Linux for quicker results:
pytest --workers auto
```


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



### Model Tracking (Shark Inference)

| Hugging Face Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :heavy_check_mark: (JIT)          | :heavy_check_mark:         |          |             |
| Albert              | :heavy_check_mark: (JIT)            | :heavy_check_mark:         |          |             |
| BigBird             | :heavy_check_mark: (AOT)            |          |          |             |
| DistilBERT          | :heavy_check_mark: (AOT)            |          |          |             |
| GPT2                | :x: (AOT)            |          |          |             |


| TORCHVISION Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|--------------------|----------------------|----------|----------|-------------|
| AlexNet            | :heavy_check_mark: (Script)         |          |          |             |
| DenseNet121        | :heavy_check_mark: (Script)         |          |          |             |
| MNasNet1_0         | :heavy_check_mark: (Script)         |          |          |             |
| MobileNetV2        | :heavy_check_mark: (Script)         |          |          |             |
| MobileNetV3        | :heavy_check_mark: (Script)         |          |          |             |
| Unet               | :x: (Script)         |          |          |             |
| Resnet18           | :heavy_check_mark: (Script)         | :heavy_check_mark:         |  :heavy_check_mark:        |             |
| Resnet50           | :heavy_check_mark: (Script)         | :heavy_check_mark:         |   :heavy_check_mark:       |             |
| Resnext50_32x4d    | :heavy_check_mark: (Script)         |          |          |             |
| ShuffleNet_v2      | :x: (Script)         |          |          |             |
| SqueezeNet         | :heavy_check_mark: (Script)         | :heavy_check_mark:         |   :heavy_check_mark:       |             |
| EfficientNet       | :heavy_check_mark: (Script)         |          |          |             |
| Regnet             | :heavy_check_mark: (Script)         |          |          |             |
| Resnest            | :x: (Script)         |          |          |             |
| Vision Transformer | :heavy_check_mark: (Script)         |          |          |             |
| VGG 16             | :heavy_check_mark: (Script)         |          |          |             |
| Wide Resnet        | :heavy_check_mark: (Script)         | :heavy_check_mark:         | :heavy_check_mark:         |             |
| RAFT               | :x: (JIT)            |          |          |             |

For more information refer to [MODEL TRACKING SHEET](https://docs.google.com/spreadsheets/d/15PcjKeHZIrB5LfDyuw7DGEEE8XnQEX2aX8lm8qbxV8A/edit#gid=0)

### Shark Trainer API

| Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :x:           | :x:         |          |             |
| FullyConnected                | :heavy_check_mark:           | :heavy_check_mark:         |          |             |


#### Related Project Channels

*   [Upstream IREE issues](https://github.com/google/iree/issues): Feature requests,
    bugs, and other work tracking
*   [Upstream IREE Discord server](https://discord.gg/26P4xW4): Daily development
    discussions with the core team and collaborators
*   [iree-discuss email list](https://groups.google.com/forum/#!forum/iree-discuss):
    Announcements, general and low-priority discussion
*   [MLIR topic within LLVM Discourse](https://llvm.discourse.group/c/llvm-project/mlir/31):
    IREE is enabled by and heavily relies on [MLIR](https://mlir.llvm.org). IREE
    sometimes is referred to in certain MLIR discussions. Useful if you are also
    interested in MLIR evolution.
    
    
## License

nod.ai SHARK is licensed under the terms of the Apache 2.0 License with LLVM Exceptions.
See [LICENSE](LICENSE) for more information.
