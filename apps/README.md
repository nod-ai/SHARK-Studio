# [WIP] ControlNet integration:
* Since this is dependent on `diffusers` and currently there's a [WIP draft PR](https://github.com/huggingface/diffusers/pull/2407), we'd need to clone [takuma104/diffusers](https://github.com/takuma104/diffusers/), checkout `controlnet` [branch](https://github.com/takuma104/diffusers/tree/controlnet) and build `diffusers` as mentioned [here](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md).
* NOTE: Run `pip uninstall diffusers` before building it from scratch in the previous step.
* Currently we've included `ControlNet` (Canny feature) as part of our SharkStableDiffusion pipeline.
* To test it you'd first need to download [control_sd15_canny.pth](https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_canny.pth) and pass it through [convert_controlnet_to_diffusers.py](https://github.com/huggingface/diffusers/blob/faf1cfbe826c88366524e92fa27b2104effdb8c4/scripts/convert_controlnet_to_diffusers.py).
* You then need to modify [this](https://github.com/nod-ai/SHARK/blob/5e5c86f4893bfdbfc2ca310803beb4ef7146213f/apps/stable_diffusion/src/pipelines/pipeline_shark_stable_diffusion_utils.py#L62) line to point to the extracted `diffusers` checkpoint directory.
* Download [sample image](https://drive.google.com/file/d/13ncuKjTO-reK-nBcLM1Lzuli5WPPB38v/view?usp=sharing) and provide the path [here](https://github.com/nod-ai/SHARK/blob/5e5c86f4893bfdbfc2ca310803beb4ef7146213f/apps/stable_diffusion/src/pipelines/pipeline_shark_stable_diffusion_txt2img.py#L152).
* Finally run 
```shell
python apps/stable_diffusion/scripts/txt2img.py --precision=fp32 --device=cuda --prompt="bird" --max_length=64 --import_mlir --enable_stack_trace --no-use_tuned --hf_model_id="runwayml/stable-diffusion-v1-5" --scheduler="DDIM"
```