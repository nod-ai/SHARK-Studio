# Stable Diffusion optimized for AMD RDNA2/RDNA3 GPUs

Before you start, please be aware that this is beta software that relies on a special AMD driver. Like all StableDiffusion GUIs published so far, you need some technical expertise to set it up. We apologize in advance if you bump into issues. If that happens, please don't hesitate to ask our Discord community for help! Please be assured that we (Nod and AMD) are working hard to improve the user experience in coming months.
If it works well for you, please "star" the following GitHub projects... this is one of the best ways to help and spread the word!

* https://github.com/nod-ai/SHARK
* https://github.com/iree-org/iree

## Install this latest AMD Drivers

### AMD KB Drivers for RDNA2 and RDNA3:

*AMD Software: (Adrenalin Edition 23.2.1) [https://www.amd.com/en/support/kb/release-notes/rn-rad-win-23-2-1] 

## Installation

Download the latest Windows SHARK SD binary [492 here](https://github.com/nod-ai/SHARK/releases/download/20230203.492/shark_sd_20230203_492.exe) in a folder of your choice. If you want nighly builds, you can look for them on the GitHub releases page.

Notes:
* We recommend that you download this EXE in a new folder, whenever you download a new EXE version. If you download it in the same folder as a previous install, you must delete the old `*.vmfb` files. Those contain Vulkan dispatches compiled from MLIR which can be outdated if you run a new EXE from the same folder. You can use `--clear_all` flag once to clean all the old files. 
* If you recently updated the driver or this binary (EXE file), we recommend you:
  * clear all the local artifacts with `--clear_all` OR 
  * clear the Vulkan shader cache: For Windows users this can be done by clearing the contents of `C:\Users\%username%\AppData\Local\AMD\VkCache\`. On Linux the same cache is typically located at `~/.cache/AMD/VkCache/`.
  * clear the `huggingface` cache. In Windows, this is `C:\Users\%username%\.cache\huggingface`.

## Running

* Open a Command Prompt or Powershell terminal, change folder (`cd`) to the .exe folder. Then run the EXE from the command prompt. That way, if an error occurs, you'll be able to cut-and-paste it to ask for help. (if it always works for you without error, you may simply double-click the EXE to start the web browser)
* The first run may take few minutes when the models are downloaded and compiled. Your patience is appreciated. The download could be about 5GB.
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


The output on a 7900XTX would like:

```shell 
Stats for run 0:
Average step time: 47.19188690185547ms/it
Clip Inference time (ms) = 109.531
VAE Inference time (ms): 78.590

Total image generation time: 2.5788655281066895sec
```

Find us on [SHARK Discord server](https://discord.gg/RUqY2h2s9u) if you have any trouble with running it on your hardware. 
