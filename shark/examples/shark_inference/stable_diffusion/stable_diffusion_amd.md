# Stable Diffusion optimized for AMD RDNA2/RDNA3 GPUs

## Install the latest AMD Drivers

### RDNA2 Drivers:
*AMD Software: Adrenalin Edition 22.11.1 for MLIR/IREE Driver Version 22.20.29.09 for Windows® 10 and Windows® 11 (Windows Driver Store Version 31.0.12029.9003)*

https://www.amd.com/en/support/kb/release-notes/rn-rad-win-22-11-1-mril-iree

Note that if you previously tried Stable Diffusion with a different driver, it may be necessary to clear vulkan cache after changing drivers.

For Windows users this can be done by clearing the contents of `C:\Users\<username>\AppData\Local\AMD\VkCache\`. On Linux the same cache is typically located at `~/.cache/AMD/VkCache/`.

## Installation

Download the latest Windows `shark_sd.exe` [here](https://storage.googleapis.com/shark-public/anush/windows/shark_sd.exe) and also download the latest Google Cloud Storage tool `gsutil.exe` [here](https://storage.googleapis.com/shark-public/anush/windows/gsutil.exe) and place them in the same directory and double click on `stable_sd.exe`. Accept if Windows warns of an unsigned .exe. The requirement to download `gsutil.exe` will be removed in the next few days.


#### Access Stable Diffusion on http://localhost:8080/?__theme=dark


<img width="1607" alt="webui" src="https://user-images.githubusercontent.com/74956/204939260-b8308bc2-8dc4-47f6-9ac0-f60b66edab99.png">


Here are some samples generated:

![tajmahal, snow, sunflowers, oil on canvas_0](https://user-images.githubusercontent.com/74956/204934186-141f7e43-6eb2-4e89-a99c-4704d20444b3.jpg)

![a photo of a crab playing a trumpet](https://user-images.githubusercontent.com/74956/204933258-252e7240-8548-45f7-8253-97647d38313d.jpg)


<details>
  <summary>Advanced Installation </summary>

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



### Run Stable Diffusion on your device - Commandline

#### Windows 10/11 Users
```powershell
(shark.venv) PS C:\g\shark> python .\shark\examples\shark_inference\stable_diffusion\main.py --precision="fp16" --prompt="tajmahal, snow, sunflowers, oil on canvas" --device="vulkan"
```

#### Linux
```shell
python3.10 shark/examples/shark_inference/stable_diffusion/main.py --precision=fp16 --device=vulkan --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd"
```

The output on a 6900XT would like:

```shell 
44it [00:08,  5.14it/s]i = 44 t = 120 (191ms)
45it [00:08,  5.15it/s]i = 45 t = 100 (191ms)
46it [00:08,  5.16it/s]i = 46 t = 80 (191ms)
47it [00:09,  5.16it/s]i = 47 t = 60 (193ms)
48it [00:09,  5.15it/s]i = 48 t = 40 (195ms)
49it [00:09,  5.12it/s]i = 49 t = 20 (196ms)
50it [00:09,  5.14it/s]
Average step time: 192.8154182434082ms/it
Total image generation runtime (s): 10.390909433364868
(shark.venv) PS C:\g\shark>
```


For more options to the Stable Diffusion model read [this](https://github.com/nod-ai/SHARK/blob/main/shark/examples/shark_inference/stable_diffusion/README.md)

<details>

Find us on [SHARK Discord server](https://discord.gg/RUqY2h2s9u) if you have any trouble with running it on your hardware. 
