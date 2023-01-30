# Stable Diffusion optimized for AMD RDNA2/RDNA3 GPUs

Before you start, please be aware that this is beta software that relies on a special AMD driver. Like all StableDiffusion GUIs published so far, you need some technical expertise to set it up. We apologize in advance if you bump into issues. If that happens, please don't hesitate to ask our Discord community for help! If you still can't get it to work, we're sorry, and please be assured that we (Nod and AMD) are working hard to improve the user experience in coming months.
If it works well for you, please "star" the following GitHub projects... this is one of the best ways to help and spread the word!

* https://github.com/nod-ai/SHARK
* https://github.com/iree-org/iree

## Install this specific AMD Drivers (AMD latest may not have all the fixes).

### AMD KB Drivers for RDNA2 and RDNA3:

*AMD Software: Adrenalin Edition 22.11.1 for MLIR/IREE Driver Version 22.20.29.09 for Windows® 10 and Windows® 11 (Windows Driver Store Version 31.0.12029.9003)*

First, for RDNA2 users, download this special driver in a folder of your choice. We recommend you keep the installation files around, since you may need to re-install it later, if Windows Update decides to overwrite it:
https://www.amd.com/en/support/kb/release-notes/rn-rad-win-22-11-1-mlir-iree

For RDNA3, the latest driver 23.1.2 supports MLIR/IREE as well: https://www.amd.com/en/support/kb/release-notes/rn-rad-win-23-1-2-kb

KNOWN ISSUES with this special AMD driver:
* `Windows Update` may (depending how it's configured) automatically install a new official AMD driver that overwrites this IREE-specific driver. If Stable Diffusion used to work, then a few days later, it slows down a lot or produces incorrect results (e.g. black images), this may be the cause. To fix this problem, please check the installed driver version, and re-install the special driver if needed. (TODO: document how to prevent this `Windows Update` behavior!)
* Some people using this special driver experience mouse pointer accuracy issues, especially if using a larger-than-default mouse pointer. The clicked point isn't centered properly. One possible work-around is to reset the pointer size to "1" in "Change pointer size and color".

## Installation

Download the latest Windows SHARK SD binary [469 here](https://github.com/nod-ai/SHARK/releases/download/20230124.469/shark_sd_20230124_469.exe) in a folder of your choice. If you want nighly builds, you can look for them on the GitHub releases page.

Notes:
* We recommend that you download this EXE in a new folder, whenever you download a new EXE version. If you download it in the same folder as a previous install, you must delete the old `*.vmfb` files. Those contain Vulkan dispatches compiled from MLIR which can be outdated if you run a new EXE from the same folder. You can use `--clean_all` flag once to clean all the old files. 
* If you recently updated the driver or this binary (EXE file), we recommend you:
  * clear all the local artifacts with `--clear_all` OR 
  * clear the Vulkan shader cache: For Windows users this can be done by clearing the contents of `C:\Users\%username%\AppData\Local\AMD\VkCache\`. On Linux the same cache is typically located at `~/.cache/AMD/VkCache/`.
  * clear the `huggingface` cache. In Windows, this is `C:\Users\%username%\.cache\huggingface`.

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


## Setup your Python Virtual Environment and Dependencies
<details>
 <summary> Windows 10/11 Users </summary>

* Install the latest Python 3.10.x version from [here](https://www.python.org/downloads/windows/)

* Install Git for Windows from [here](https://git-scm.com/download/win)
  
* Install Rust for Windows from [here](https://www.rust-lang.org/tools/install)

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
</details> 

 <details>
  <summary>Linux</summary>

```shell
git clone https://github.com/nod-ai/SHARK.git
cd SHARK
./setup_venv.sh
source shark.venv/bin/activate
```
 </details>

### Run Stable Diffusion on your device - WebUI

<details>
 <summary>Windows 10/11 Users</summary>
 
```powershell
(shark.venv) PS C:\Users\nod\SHARK> cd web
(shark.venv) PS C:\Users\nod\SHARK\web> python index.py
```
 
 </details>
 
<details>
 <summary>Linux Users</summary>
 
```shell
(shark.venv) > cd web
(shark.venv) > python index.py
```
 
</details>

### Run Stable Diffusion on your device - Commandline

<details>
 <summary>Windows 10/11 Users</summary>
 
```powershell
(shark.venv) PS C:\g\shark> python .\shark\examples\shark_inference\stable_diffusion\main.py --precision="fp16" --prompt="tajmahal, snow, sunflowers, oil on canvas" --device="vulkan"
```
 
  </details>

<details>
 <summary>Linux</summary>
 
```shell
python3.10 shark/examples/shark_inference/stable_diffusion/main.py --precision=fp16 --device=vulkan --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd"
```
 
  </details>

The output on a 7900XTX would like:

```shell 
Stats for run 0:
Average step time: 47.19188690185547ms/it
Clip Inference time (ms) = 109.531
VAE Inference time (ms): 78.590

Total image generation time: 2.5788655281066895sec
```

For more options to the Stable Diffusion model read [this](https://github.com/nod-ai/SHARK/blob/main/shark/examples/shark_inference/stable_diffusion/README.md)
 
</details>
  <details>
  <summary>Discord link</summary>
Find us on [SHARK Discord server](https://discord.gg/RUqY2h2s9u) if you have any trouble with running it on your hardware. 
</details>
