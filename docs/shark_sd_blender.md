# Overview

This document is intended to provide a starting point for using SHARK stable diffusion with Blender. 

We currently make use of the [AI-Render Plugin](https://github.com/benrugg/AI-Render) to integrate with Blender.

## Setup SHARK and prerequisites:

 * Download the latest SHARK SD webui .exe from [here](https://github.com/nod-ai/SHARK/releases) or follow instructions on the [README](https://github.com/nod-ai/SHARK#readme)
 * Once you have the .exe where you would like SHARK to install, run the .exe from terminal/PowerShell with the `--api` flag:
```
## Run the .exe in API mode:
.\shark_sd_<date>_<ver>.exe --api

## For example:
.\shark_sd_20230411_671.exe --api --server_port=8082

## From a the base directory of a source clone of SHARK:
./setup_venv.ps1
python apps\stable_diffusion\web\index.py --api

```

Your local SD server should start and look something like this:
![image](https://user-images.githubusercontent.com/87458719/231369758-e2c3c45a-eccc-4fe5-a788-4a3bf1ace1d1.png)

 * Note: When running in api mode with `--api`, the .exe will not function as a webUI. Thus, the address in the terminal output will only be useful for API requests.

### Install AI Render

- Get AI Render on [Blender Market](https://blendermarket.com/products/ai-render) or [Gumroad](https://airender.gumroad.com/l/ai-render)
- Open Blender, then go to Edit > Preferences > Add-ons > Install and then find the zip file
- We will be using the Automatic1111 SD backend for the AI-Render plugin. Follow instructions [here](https://github.com/benrugg/AI-Render/wiki/Local-Installation) to setup local SD backend.

Your AI-Render preferences should be configured as shown; the highlighted part should match your terminal output:
![image](https://user-images.githubusercontent.com/87458719/231390322-59a54a09-520a-4a08-b658-6e37bd63e932.png)


The [AI-Render README](https://github.com/benrugg/AI-Render/blob/main/README.md) has more details on installation and usage, as well as video tutorials.

## Using AI-Render + SHARK in your Blender project

- In the Render Properties tab, in the AI-Render dropdown, enable AI-Render.

![image](https://user-images.githubusercontent.com/87458719/231392843-9bd51744-3ce2-464e-843a-0c4d4c96df0c.png)

- Select an image size (it's usually better to upscale later than go high on the img2img resolution here.)

![image](https://user-images.githubusercontent.com/87458719/231394288-0c4ab8c5-dc30-4dbe-8bc1-7520ded5efe8.png)

- From here, you can enter a prompt and configure img2img Stable Diffusion parameters, and AI-Render will run SHARK SD img2img on the rendered scene.
- AI-Render has useful presets for aesthetic styles, so you should be able to keep your subject prompt simple and focus on creating a decent Blender scene to start from.

![image](https://user-images.githubusercontent.com/87458719/231440729-2fe69586-41cb-4274-9ce7-f6c08def600b.png)

## Examples:
Scene:
![blender-sample-2](https://user-images.githubusercontent.com/87458719/231450408-0e680086-3e52-4962-a5c1-c703a94d1583.png)

Prompt:
"A bowl of tangerines in front of rocks, masterpiece, oil on canvas, by Georgia O'Keefe, trending on artstation, landscape painting by Caspar David Friedrich"

Negative Prompt (default):
"ugly, bad art, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, tiling, signature, cut off, draft"

Example output:
![blender-sample-2_out](https://user-images.githubusercontent.com/87458719/231451145-a0b56897-a7d0-4add-bbed-7e8af21a65df.png)






