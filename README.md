# SHARK

High Performance Machine Learning Distribution

[![Nightly Release](https://github.com/nod-ai/SHARK/actions/workflows/nightly.yml/badge.svg)](https://github.com/nod-ai/SHARK/actions/workflows/nightly.yml)
[![Validate torch-models on Shark Runtime](https://github.com/nod-ai/SHARK/actions/workflows/test-models.yml/badge.svg)](https://github.com/nod-ai/SHARK/actions/workflows/test-models.yml)


<details>
  <summary>Prerequisites - Drivers </summary>
  
#### Install your Windows hardware drivers
* [AMD RDNA Users] Download the latest driver [here](https://www.amd.com/en/support/kb/release-notes/rn-rad-win-23-2-1).
* [macOS Users] Download and install the 1.3.216 Vulkan SDK from [here](https://sdk.lunarg.com/sdk/download/1.3.216.0/mac/vulkansdk-macos-1.3.216.0.dmg). Newer versions of the SDK will not work. 
* [Nvidia Users] Download and install the latest CUDA / Vulkan drivers from [here](https://developer.nvidia.com/cuda-downloads)
  
#### Linux Drivers
* MESA / RADV drivers wont work with FP16. Please use the latest AMGPU-PRO drivers (non-pro OSS drivers also wont work) or the latest NVidia Linux Drivers.

Other users please ensure you have your latest vendor drivers and Vulkan SDK from [here](https://vulkan.lunarg.com/sdk/home) and if you are using vulkan check `vulkaninfo` works in a terminal window

</details>


 
### Quick Start for SHARK Stable Diffusion for Windows 10/11 Users

Install Driver from [Prerequisites](https://github.com/nod-ai/SHARK#install-your-hardware-drivers) above 

Download the latest .exe https://github.com/nod-ai/SHARK/releases. 

Double click the .exe and you should have the [UI]( http://localhost:8080/?__theme=dark) in the browser. 

If you have custom models (ckpt, safetensors) put in a `models/` directory where the .exe is. 

Enjoy. 

Some known AMD Driver quirks and are documented [here](https://github.com/nod-ai/SHARK/blob/main/apps/stable_diffusion/stable_diffusion_amd.md ).


<details>
  <summary>Advanced Installation (Only for developers)</summary>
  
## Advanced Installation (Windows, Linux and macOS) for developers

## Check out the code

```shell
git clone https://github.com/nod-ai/SHARK.git
cd SHARK
```

## Setup your Python VirtualEnvironment and Dependencies

### Windows 10/11 Users

* Install the latest Python 3.10.x version from [here](https://www.python.org/downloads/windows/)

* Install Git for Windows from [here](https://git-scm.com/download/win)

#### Allow the install script to run in Powershell
```powershell
set-executionpolicy remotesigned
```

#### Setup venv and install necessary packages (torch-mlir, nodLabs/Shark, ...)
```powershell
./setup_venv.ps1 #You can re-run this script to get the latest version
```

### Linux / macOS Users

```shell
./setup_venv.sh
source shark.venv/bin/activate
```


### Run Stable Diffusion on your device - WebUI

#### Windows 10/11 Users
```powershell
(shark.venv) PS C:\g\shark> cd .\apps\stable_diffusion\web\
(shark.venv) PS C:\g\shark\apps\stable_diffusion\web> python .\index.py
```
#### Linux / macOS Users
```shell
(shark.venv) > cd apps/stable_diffusion/web
(shark.venv) > python index.py
```

#### Access Stable Diffusion on http://localhost:8080/?__theme=dark


<img width="1607" alt="webui" src="https://user-images.githubusercontent.com/74956/204939260-b8308bc2-8dc4-47f6-9ac0-f60b66edab99.png">



### Run Stable Diffusion on your device - Commandline

#### Windows 10/11 Users
```powershell
(shark.venv) PS C:\g\shark> python .\apps\stable_diffusion\scripts\txt2img.py --precision="fp16" --prompt="tajmahal, snow, sunflowers, oil on canvas" --device="vulkan"
```

#### Linux / macOS Users
```shell
python3.10 apps/stable_diffusion/scripts/txt2img.py --precision=fp16 --device=vulkan --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd"
```

You can replace `vulkan` with `cpu` to run on your CPU or with `cuda` to run on CUDA devices. If you have multiple vulkan devices you can address them with `--device=vulkan://1` etc
</details>

The output on a 7900XTX would like:

```shell 
Stats for run 0:
Average step time: 47.19188690185547ms/it
Clip Inference time (ms) = 109.531
VAE Inference time (ms): 78.590

Total image generation time: 2.5788655281066895sec
```

Here are some samples generated:

![tajmahal, snow, sunflowers, oil on canvas_0](https://user-images.githubusercontent.com/74956/204934186-141f7e43-6eb2-4e89-a99c-4704d20444b3.jpg)

![a photo of a crab playing a trumpet](https://user-images.githubusercontent.com/74956/204933258-252e7240-8548-45f7-8253-97647d38313d.jpg)


Find us on [SHARK Discord server](https://discord.gg/RUqY2h2s9u) if you have any trouble with running it on your hardware. 


<details>
  <summary>Binary Installation</summary>

### Setup a new pip Virtual Environment

This step sets up a new VirtualEnv for Python

```shell
python --version #Check you have 3.10 on Linux, macOS or Windows Powershell
python -m venv shark_venv
source shark_venv/bin/activate   # Use shark_venv/Scripts/activate on Windows

# If you are using conda create and activate a new conda env

# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
```

*macOS Metal* users please install https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.dmg and enable "System wide install"

### Install SHARK

This step pip installs SHARK and related packages on Linux Python 3.7, 3.8, 3.9, 3.10 and macOS Python 3.10

```shell
pip install nodai-shark -f https://nod-ai.github.io/SHARK/package-index/ -f https://llvm.github.io/torch-mlir/package-index/ -f  https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

### Run shark tank model tests.
```shell
pytest tank/test_models.py
```
See tank/README.md for a more detailed walkthrough of our pytest suite and CLI.

### Download and run Resnet50 sample

```shell
curl -O https://raw.githubusercontent.com/nod-ai/SHARK/main/shark/examples/shark_inference/resnet50_script.py
#Install deps for test script
pip install --pre torch torchvision torchaudio tqdm pillow gsutil --extra-index-url https://download.pytorch.org/whl/nightly/cpu
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
  <summary>Development, Testing and Benchmarks</summary>

If you want to use Python3.10 and with TF Import tools you can use the environment variables like:
Set `USE_IREE=1` to use upstream IREE
```
# PYTHON=python3.10 VENV_DIR=0617_venv IMPORTER=1 ./setup_venv.sh 
```

### Run any of the hundreds of SHARK tank models via the test framework
```shell
python -m  shark.examples.shark_inference.resnet50_script --device="cpu" # Use gpu | vulkan
# Or a pytest
pytest tank/test_models.py -k "MiniLM"
```
  

If you are a *Torch-mlir developer or an IREE developer* and want to test local changes you can uninstall
the provided packages with `pip uninstall torch-mlir` and / or `pip uninstall iree-compiler iree-runtime` and build locally
with Python bindings and set your PYTHONPATH as mentioned [here](https://github.com/iree-org/iree/tree/main/docs/api_docs/python#install-iree-binaries)
for IREE and [here](https://github.com/llvm/torch-mlir/blob/main/development.md#setup-python-environment-to-export-the-built-python-packages)
for Torch-MLIR.

### How to use your locally built Torch-MLIR with SHARK
```shell
1.) Run `./setup_venv.sh in SHARK` and activate `shark.venv` virtual env.
2.) Run `pip uninstall torch-mlir`.
3.) Go to your local Torch-MLIR directory.
4.) Activate mlir_venv virtual envirnoment.
5.) Run `pip uninstall -r requirements.txt`.
6.) Run `pip install -r requirements.txt`.
7.) Build Torch-MLIR.
8.) Activate shark.venv virtual environment from the Torch-MLIR directory.
8.) Run `export PYTHONPATH=`pwd`/build/tools/torch-mlir/python_packages/torch_mlir:`pwd`/examples` in the Torch-MLIR directory.
9.) Go to the SHARK directory.
```
Now the SHARK will use your locally build Torch-MLIR repo.


## Benchmarking Dispatches

To produce benchmarks of individual dispatches, you can add `--dispatch_benchmarks=All --dispatch_benchmarks_dir=<output_dir>` to your command line argument.  
If you only want to compile specific dispatches, you can specify them with a space seperated string instead of `"All"`.  E.G. `--dispatch_benchmarks="0 1 2 10"`

if you want to instead incorporate this into a python script, you can pass the `dispatch_benchmarks` and `dispatch_benchmarks_dir` commands when initializing `SharkInference`, and the benchmarks will be generated when compiled.  E.G:

```
shark_module = SharkInference(
        mlir_model,
        func_name,
        device=args.device,
        mlir_dialect="tm_tensor",
        dispatch_benchmarks="all",
        dispatch_benchmarks_dir="results"
    )
```

Output will include:
- An ordered list ordered-dispatches.txt of all the dispatches with their runtime
- Inside the specified directory, there will be a directory for each dispatch (there will be mlir files for all dispatches, but only compiled binaries and benchmark data for the specified dispatches)
- An .mlir file containing the dispatch benchmark 
- A compiled .vmfb file containing the dispatch benchmark
- An .mlir file containing just the hal executable
- A compiled .vmfb file of the hal executable
- A .txt file containing benchmark output


See tank/README.md for instructions on how to run model tests and benchmarks from the SHARK tank.

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

SHARK is maintained to support the latest innovations in ML Models: 

| TF HuggingFace Models | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------|----------|-------------|
| BERT                | :green_heart:         | :green_heart:         | :green_heart:            |
| DistilBERT         | :green_heart:         | :green_heart:         | :green_heart:            |
| GPT2         | :green_heart:         | :green_heart:         | :green_heart:            |
| BLOOM         | :green_heart:         | :green_heart:         | :green_heart:            |
| Stable Diffusion         | :green_heart:         | :green_heart:         | :green_heart:            |
| Vision Transformer       | :green_heart:         | :green_heart:         | :green_heart:            |
| ResNet50         | :green_heart:         | :green_heart:         | :green_heart:            |

For a complete list of the models supported in SHARK, please refer to [tank/README.md](https://github.com/nod-ai/SHARK/blob/main/tank/README.md).

## Communication Channels

*   [SHARK Discord server](https://discord.gg/RUqY2h2s9u): Real time discussions with the SHARK team and other users
*   [GitHub issues](https://github.com/nod-ai/SHARK/issues): Feature requests, bugs etc

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
