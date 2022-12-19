# Stable Diffusion optimized for AMD RDNA2/RDNA3 GPUs

Before you start, please be aware that this is beta software that relies on a special AMD driver. Like all StableDiffusion GUIs published so far, you need some technical expertise to set it up. We apologize in advance if you bump into issues. If that happens, please don't hesitate to ask our Discord community for help! If you still can't get it to work, we're sorry, and please be assured that we (Nod and AMD) are working hard to improve the user experience in coming months.
If it works well for you, please "star" the following GitHub projects... this is one of the best ways to help and spread the word!

* https://github.com/nod-ai/SHARK
* https://github.com/iree-org/iree

## Install the latest AMD Drivers

Please note that the **RDNA3 GPUs aren't supported yet**, but will be very soon. Thank you very much for your patience!

### RDNA2 KB Drivers:

*AMD Software: Adrenalin Edition 22.11.1 for MLIR/IREE Driver Version 22.20.29.09 for Windows® 10 and Windows® 11 (Windows Driver Store Version 31.0.12029.9003)*

First, download this special driver in a folder of your choice. We recommend you keep that driver around since you may need to re-install it later, if Windows Update decides to overwrite it:
https://www.amd.com/en/support/kb/release-notes/rn-rad-win-22-11-1-mlir-iree

KNOWN ISSUES with this special AMD driver:
* `Windows Update` may (depending how it's configured) automatically install a new official AMD driver that overwrites this IREE-specific driver. If Stable Diffusion used to work, then a few days later, it slows down a lot or produces incorrect results (e.g. black images), this may be the cause. To fix this problem, please check the installed driver's version, and re-install the special driver if needed. (TODO: document how to prevent this `Windows Update` behavior!)
* Some people using this special driver experience mouse pointer accuracy issues, if you use a larger-than-default mouse pointer. The clicked point isn't centered properly. One possible work-around is to reset the pointer size to "1" in "Change pointer size and color".

## Installation

Download the latest Windows SHARK SD binary [here](https://github.com/nod-ai/SHARK/releases/download/20221219.398/shark_sd_20221219_398.exe) in a folder of your choice. Please read carefully the following notes:

Notes:
* We recommend that you download this EXE in a new folder, whenever you download a new EXE version. If you download it in the same folder as a previous install, you must delete the old `*.vmfb` files. Those contain Vulkan dispatches compiled from MLIR, that can get outdated if you run multiple EXE from the same folder.
* Your browser may warn you about downloading an .exe file
* If you recently updated the driver or this binary (EXE file), we recommend you:
  * clear the Vulkan shader cache: For Windows users this can be done by clearing the contents of `C:\Users\<username>\AppData\Local\AMD\VkCache\`. On Linux the same cache is typically located at `~/.cache/AMD/VkCache/`.
  * clear the `huggingface` cache. In Windows, this is `C:\Users\<username>\.cache\huggingface`.

## Running

* Open a Command Prompt or Powershell terminal, change folder (`cd`) to the .exe folder. Then run the EXE from the command prompt. That way, if an error occurs, you'll be able to cut-and-paste it to ask for help. (if it always works for you without error, you may simply double-click the EXE to start the web browser)
* The first run may take about 10-15 minutes when the models are downloaded and compiled. Your patience is appreciated. The download could be about 5GB.
* If successful, you will likely see a Windows Defender message asking you to give permission to open a web server port. Accept it.
* Open a browser to access the Stable Diffusion web server. By default, the port is 8080, so you can go to http://localhost:8080/?__theme=dark.

## Stopping

* Select the command prompt that's running the EXE. Press CTRL-C and wait a moment. The application should stop. 
* Please make sure to do the above step before you attempt to update the EXE to a new version.

# Results

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
</details>
<details>
  <summary>Discord link</summary>
Find us on [SHARK Discord server](https://discord.gg/RUqY2h2s9u) if you have any trouble with running it on your hardware. 
</details>
