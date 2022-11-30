# Stable Diffusion optimized for AMD RDNA2/RDNA3 GPUs

## Install the latest AMD Drivers

### RDNA2 Drivers:
*AMD Software: Adrenalin Edition 22.11.1 for MLIR/IREE Driver Version 22.20.29.09 for Windows® 10 and Windows® 11 (Windows Driver Store Version 31.0.12029.9003)*

https://www.amd.com/en/support/kb/release-notes/rn-rad-win-22-11-1-mril-iree


## Installation (Windows, Linux)

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
git clone https://github.com/nod-ai/SHARK.git
cd SHARK
./setup_venv.ps1 #You can re-run this script to get the latest version
```

### Linux

```shell
git clone https://github.com/nod-ai/SHARK.git
cd SHARK
./setup_venv.sh
source shark.venv/bin/activate
```

### Run Stable Diffusion on your device - WebUI

#### Windows 10/11 Users
```powershell
(shark.venv) PS C:\Users\nod\SHARK> cd web
(shark.venv) PS C:\Users\nod\SHARK\web> python index.py
```
#### Linux Users
```shell
(shark.venv) > cd web
(shark.venv) > python index.py
```

Access Stable Diffusion on http://localhost:8080


### Run Stable Diffusion on your device - Commandline

#### Windows 10/11 Users
```powershell
(shark.venv) PS C:\g\shark> python .\shark\examples\shark_inference\stable_diffusion\main.py --precision="fp16" --prompt="tajmahal, snow, sunflowers, oil on canvas" --device="vulkan"
```

#### Linux
```shell
python3.10 shark/examples/shark_inference/stable_diffusion/main.py --precision=fp16 --device=vulkan --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd"
```

Here are some samples generated:

![tajmahal, snow, sunflowers, oil on canvas_0](https://user-images.githubusercontent.com/74956/204933218-e8e8e16e-d83f-4384-83f5-13647b6e35ce.jpg)

![a photo of a crab playing a trumpet](https://user-images.githubusercontent.com/74956/204933258-252e7240-8548-45f7-8253-97647d38313d.jpg)


For more options to the Stable Diffusion model read [this](https://github.com/nod-ai/SHARK/blob/main/shark/examples/shark_inference/stable_diffusion/README.md)



Find us on [SHARK Discord server](https://discord.gg/RUqY2h2s9u) if you have any trouble with running it on your hardware. 
