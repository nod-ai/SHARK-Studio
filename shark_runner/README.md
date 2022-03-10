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
