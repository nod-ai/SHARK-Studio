# Shark Runner

The Shark Runner provides inference and training APIs to run deep learning models on Shark Runtime.

# How to configure.

## Check out the code

```shell
git clone https://github.com/NodLabs/dSHARK.git 
```

## Setup your Python VirtualEnvironment and Dependencies
```shell
# Setup venv and install necessary packages (torch-mlir, nodLabs/Shark, ...).
./setup_venv.sh
# Please activate the venv after installation.
```

### Run a demo script
```shell
python -m  shark.examples.resnet50_script
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

| Hugging Face Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :heavy_check_mark: (JIT)          |          |          |             |
| Albert              | :heavy_check_mark: (JIT)            |          |          |             |
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
| Resnet18           | :heavy_check_mark: (Script)         | :heavy_check_mark:         |          |             |
| Resnet50           | :heavy_check_mark: (Script)         | :heavy_check_mark:         |          |             |
| Resnext50_32x4d    | :heavy_check_mark: (Script)         |          |          |             |
| ShuffleNet_v2      | :x: (Script)         |          |          |             |
| SqueezeNet         | :x: (Script)         |          |          |             |
| EfficientNet       | :heavy_check_mark: (Script)         |          |          |             |
| Regnet             | :heavy_check_mark: (Script)         |          |          |             |
| Resnest            | :x: (Script)         |          |          |             |
| Vision Transformer | :heavy_check_mark: (Script)         |          |          |             |
| VGG 16             | :heavy_check_mark: (Script)         |          |          |             |
| Wide Resnet        | :heavy_check_mark: (Script)         | :heavy_check_mark:         |          |             |
| RAFT               | :x: (JIT)            |          |          |             |

For more information refer to [MODEL TRACKING SHEET](https://docs.google.com/spreadsheets/d/15PcjKeHZIrB5LfDyuw7DGEEE8XnQEX2aX8lm8qbxV8A/edit#gid=0)
