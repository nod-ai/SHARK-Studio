# Shark Runner

The Shark Runner provides inference and training APIs to run deep learning models on Shark Runtime.

# How to configure.

### Build [torch-mlir](https://github.com/llvm/torch-mlir) and [iree](https://github.com/google/iree) including [iree python bindings](https://google.github.io/iree/building-from-source/python-bindings-and-importers/#using-the-python-bindings)

### Setup Python Environment
```shell
#Activate your virtual environment.
export TORCH_MLIR_BUILD_DIR=/path/to/torch-mlir/build
export IREE_BUILD_DIR=/path/to/iree-build
source set_dep_pypaths.sh
```

### Run a demo script
```shell
python resnet50.py
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
| GPT2                | :heavy_check_mark: (AOT)            |          |          |             |


| TORCHVISION Models | Torch-MLIR lowerable | IREE-CPU | IREE-GPU | IREE-VULKAN |
|--------------------|----------------------|----------|----------|-------------|
| AlexNet            | :heavy_check_mark: (Script)         |          |          |             |
| DenseNet121        | :heavy_check_mark: (Script)         |          |          |             |
| MNasNet1_0         | :heavy_check_mark: (Script)         |          |          |             |
| MobileNetV2        | :heavy_check_mark: (Script)         |          |          |             |
| MobileNetV3        | :heavy_check_mark: (Script)         |          |          |             |
| Unet               | :heavy_check_mark: (Script)         |          |          |             |
| Resnet18           | :heavy_check_mark: (Script)         |          |          |             |
| Resnet50           | :heavy_check_mark: (Script)         |          |          |             |
| Resnext50_32x4d    | :heavy_check_mark: (Script)         |          |          |             |
| ShuffleNet_v2      | :heavy_check_mark: (Script)         |          |          |             |
| SqueezeNet         | :heavy_check_mark: (Script)         |          |          |             |
| EfficientNet       | :heavy_check_mark: (Script)         |          |          |             |
| Regnet             | :heavy_check_mark: (Script)         |          |          |             |
| Resnest            | :heavy_check_mark: (Script)         |          |          |             |
| Vision Transformer | :heavy_check_mark: (Script)         |          |          |             |
| VGG 16             | :heavy_check_mark: (Script)         |          |          |             |
| Wide Resnet        | :heavy_check_mark: (Script)         |          |          |             |
| RAFT               | :heavy_check_mark: (JIT)            |          |          |             |

For more information refer to [MODEL TRACKING SHEET](https://docs.google.com/spreadsheets/d/15PcjKeHZIrB5LfDyuw7DGEEE8XnQEX2aX8lm8qbxV8A/edit#gid=0)
