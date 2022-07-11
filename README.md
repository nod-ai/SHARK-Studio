# SHARK

High Performance Machine Learning and Data Analytics for CPUs, GPUs, Accelerators and Heterogeneous Clusters

[![Nightly Release](https://github.com/nod-ai/SHARK/actions/workflows/nightly.yml/badge.svg)](https://github.com/nod-ai/SHARK/actions/workflows/nightly.yml)
[![Validate torch-models on Shark Runtime](https://github.com/nod-ai/SHARK/actions/workflows/test-models.yml/badge.svg)](https://github.com/nod-ai/SHARK/actions/workflows/test-models.yml)

## Communication Channels

*   [SHARK Discord server](https://discord.gg/RUqY2h2s9u): Real time discussions with the SHARK team and other users
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

*macOS Metal* users please install https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.dmg and enable "System wide install"

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
source shark.venv/bin/activate
```
For example if you want to use Python3.10 and upstream IREE with TF Import tools you can use the environment variables like:
```
# PYTHON=python3.10 VENV_DIR=0617_venv IMPORTER=1 USE_IREE=1 ./setup_venv.sh 
```

If you are a Torch-mlir developer or an IREE developer and want to test local changes you can uninstall
the provided packages with `pip uninstall torch-mlir` and / or `pip uninstall iree-compiler iree-runtime` and build locally
with Python bindings and set your PYTHONPATH as mentioned [here](https://google.github.io/iree/bindings/python/)
for IREE and [here](https://github.com/llvm/torch-mlir/blob/main/development.md#setup-python-environment-to-export-the-built-python-packages)
for Torch-MLIR.

### Run a demo script
```shell
python -m  shark.examples.shark_inference.resnet50_script --device="cpu" # Use gpu | vulkan
# Or a pytest
pytest tank/tf/hf_masked_lm/albert-base-v2_test.py::AlbertBaseModuleTest::test_module_static_cpu
```




</details>


<details>
  <summary>Testing</summary>

### Run all model tests on CPU/GPU/VULKAN/Metal
```shell
pytest tank

# If on Linux for quicker results:
pytest tank -n auto
```

### Running specific tests
```shell
# Run tests for a specific model:
pytest tank/<MODEL_NAME> #i.e., pytest tank/bert-base-uncased

# Run tests for a specific case:
pytest tank/<MODEL_NAME>/<MODEL_TEST>.py::<MODEL>ModuleTest::<CASE>
# i.e., pytest tank/bert-base-uncased/bert-base-uncased_test.py::BertModuleTest::test_module_static_cpu
# For frontends other than pytorch, if available for a model, add frontend to filename: tank/bert-base-uncased/bert-base-uncased_tf_test.py

# Run all tests, including tests for benchmarking and SHARK modules:
# From base SHARK directory,
pytest
```

### Run all model benchmark tests on CPU/GPU/VULKAN/Metal
```shell
pytest benchmarks
```
</details>


<details>
  <summary>API Reference</summary>

### Shark Inference API

```

from shark.shark_importer import SharkImporter

# SharkImporter imports mlir file from the torch, tensorflow or tf-lite module.

mlir_importer = SharkImporter(
    torch_module,
    (input),
    frontend="torch",  #tf, #tf-lite
)
torch_mlir, func_name = mlir_importer.import_mlir(tracing_required=True)

# SharkInference accepts mlir in linalg, mhlo, and tosa dialect.

from shark.shark_inference import SharkInference
shark_module = SharkInference(torch_mlir, func_name, device="cpu", mlir_dialect="linalg")
shark_module.compile()
result = shark_module.forward((input))

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
shark_module = SharkInference(mhlo_ir, func_name="forward", device="cpu", mlir_dialect="mhlo")
shark_module.compile()
result = shark_module.forward((arg0, arg1))
```
</details>


## Supported and Validated Models

<details>
  <summary>PyTorch Models</summary>

### Huggingface PyTorch Models

| Hugging Face Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :green_heart: (JIT)          | :green_heart:         | :green_heart:         | :green_heart:            |
| Albert              | :green_heart: (JIT)            | :green_heart:         | :green_heart:         | :green_heart:            |
| BigBird             | :green_heart: (AOT)            |          |          |             |
| DistilBERT          | :green_heart: (JIT)            | :green_heart:         | :green_heart:         | :green_heart:            |
| GPT2                | :broken_heart: (AOT)            |          |          |             |
| MobileBert          | :green_heart: (JIT)            | :green_heart:         | :green_heart:         | :green_heart:            |

### Torchvision  Models

| TORCHVISION Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|--------------------|----------------------|----------|----------|-------------|
| AlexNet            | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| DenseNet121        | :green_heart: (Script)         |          |          |             |
| MNasNet1_0         | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| MobileNetV2        | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| MobileNetV3        | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| Unet               | :broken_heart: (Script)         |          |          |             |
| Resnet18           | :green_heart: (Script)         | :green_heart:         |  :green_heart:        | :green_heart:            |
| Resnet50           | :green_heart: (Script)         | :green_heart:         |   :green_heart:       | :green_heart:            |
| Resnet101           | :green_heart: (Script)         | :green_heart:         |   :green_heart:       | :green_heart:            |
| Resnext50_32x4d    | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| ShuffleNet_v2      | :broken_heart: (Script)         |          |          |             |
| SqueezeNet         | :green_heart: (Script)         | :green_heart:         |   :green_heart:       | :green_heart:            |
| EfficientNet       | :green_heart: (Script)         |          |          |             |
| Regnet             | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| Resnest            | :broken_heart: (Script)         |          |          |             |
| Vision Transformer | :green_heart: (Script)         |          |          |             |
| VGG 16             | :green_heart: (Script)         | :green_heart:         |   :green_heart:       |             |
| Wide Resnet        | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| RAFT               | :broken_heart: (JIT)            |          |          |             |

For more information refer to [MODEL TRACKING SHEET](https://docs.google.com/spreadsheets/d/15PcjKeHZIrB5LfDyuw7DGEEE8XnQEX2aX8lm8qbxV8A/edit#gid=0)

### PyTorch Training Models

| Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :broken_heart:           | :broken_heart:         |          |             |
| FullyConnected                | :green_heart:           | :green_heart:         |          |             |

</details>

<details>
  <summary>JAX Models</summary>


### JAX  Models

| Models | JAX-MHLO lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| DALL-E                | :broken_heart:           | :broken_heart:         |          |             |
| FullyConnected                | :green_heart:           | :green_heart:         |          |             |

</details>

<details>
  <summary>TFLite Models</summary>

### TFLite Models

| Models | TOSA/LinAlg  | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :broken_heart:           | :broken_heart:         |          |             |
| FullyConnected      | :green_heart:           | :green_heart:         |          |             |
| albert | :green_heart:           | :green_heart:         |          |             |
| asr_conformer | :green_heart:           | :green_heart:         |          |             |
| bird_classifier | :green_heart:           | :green_heart:         |          |             |
| cartoon_gan | :green_heart:           | :green_heart:         |          |             |
| craft_text | :green_heart:           | :green_heart:         |          |             |
| deeplab_v3 | :green_heart:           | :green_heart:         |          |             |
| densenet | :green_heart:           | :green_heart:         |          |             |
| east_text_detector | :green_heart:           | :green_heart:         |          |             |
| efficientnet_lite0_int8 | :green_heart:           | :green_heart:         |          |             |
| efficientnet | :green_heart:           | :green_heart:         |          |             |
| gpt2 | :green_heart:           | :green_heart:         |          |             |
| image_stylization | :green_heart:           | :green_heart:         |          |             |
| inception_v4 | :green_heart:           | :green_heart:         |          |             |
| inception_v4_uint8 | :green_heart:           | :green_heart:         |          |             |
| lightning_fp16 | :green_heart:           | :green_heart:         |          |             |
| lightning_i8 | :green_heart:           | :green_heart:         |          |             |
| lightning | :green_heart:           | :green_heart:         |          |             |
| magenta | :green_heart:           | :green_heart:         |          |             |
| midas | :green_heart:           | :green_heart:         |          |             |
| mirnet | :green_heart:           | :green_heart:         |          |             |
| mnasnet | :green_heart:           | :green_heart:         |          |             |
| mobilebert_edgetpu_s_float | :green_heart:           | :green_heart:         |          |             |
| mobilebert_edgetpu_s_quant | :green_heart:           | :green_heart:         |          |             |
| mobilebert | :green_heart:           | :green_heart:         |          |             |
| mobilebert_tf2_float | :green_heart:           | :green_heart:         |          |             |
| mobilebert_tf2_quant | :green_heart:           | :green_heart:         |          |             |
| mobilenet_ssd_quant | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v1 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v1_uint8 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v2_int8 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v2 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v2_uint8 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v3-large | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v3-large_uint8 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v35-int8 | :green_heart:           | :green_heart:         |          |             |
| nasnet | :green_heart:           | :green_heart:         |          |             |
| person_detect | :green_heart:           | :green_heart:         |          |             |
| posenet | :green_heart:           | :green_heart:         |          |             |
| resnet_50_int8 | :green_heart:           | :green_heart:         |          |             |
| rosetta | :green_heart:           | :green_heart:         |          |             |
| spice | :green_heart:           | :green_heart:         |          |             |
| squeezenet | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v1 | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v1_uint8 | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v2_fpnlite | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v2_fpnlite_uint8 | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v2_int8 | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v2 | :green_heart:           | :green_heart:         |          |             |
| ssd_spaghettinet_large | :green_heart:           | :green_heart:         |          |             |
| ssd_spaghettinet_large_uint8 | :green_heart:           | :green_heart:         |          |             |
| visual_wake_words_i8 | :green_heart:           | :green_heart:         |          |             |

</details>

<details>
  <summary>TF Models</summary>

### Tensorflow Models (Inference)

| Hugging Face Models | tf-mhlo lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| albert-base-v2              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| DistilBERT          | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| CamemBert                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| ConvBert              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| Deberta              |            |         |          |             |
| electra          | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| funnel              |            |         |          |             |
| layoutlm              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| longformer              |            |         |          |             |
| mobile-bert                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| remembert              |            |         |          |             |
| tapas              |            |         |          |             |
| flaubert                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| roberta                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| xlm-roberta              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| mpnet              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |

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
