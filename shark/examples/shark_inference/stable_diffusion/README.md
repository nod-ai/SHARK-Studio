# STABLE DIFFUSION

## Installation

Follow setup instructions in the main [README.md](https://github.com/nod-ai/SHARK#readme) for regular usage. 

## Debug commands and other advanced usage follows.

```shell
python main.py --precision="fp32"|"fp16" --device="cpu"|"cuda"|"vulkan" --import_mlir|--no-import_mlir --prompt "enter the text" 

```

## dump all dispatch .spv and isa using amdllpc

```shell
python main.py --precision="fp16" --device="vulkan" --iree-vulkan-target-triple=rdna3-unknown-linux --no-load_vmfb --dispatch_benchmarks="all" --dispatch_benchmarks_dir="SD_dispatches" --dump_isa
```

## Compile and save the .vmfb (using vulkan fp16 as an example):

```shell
python shark/examples/shark_inference/stable_diffusion/main.py --precision=fp16 --device=vulkan --steps=50 --save_vmfb
```

## Capture an RGP trace

```shell
python shark/examples/shark_inference/stable_diffusion/main.py --precision=fp16 --device=vulkan --steps=50 --save_vmfb --enable_rgp
```

## Run the vae module with iree-benchmark-module (NCHW, fp16, vulkan, for example):

```shell
iree-benchmark-module --module_file=/path/to/output/vmfb --entry_function=forward --device=vulkan --function_input=1x4x64x64xf16  
```

## Run the unet module with iree-benchmark-module (same config as above):
```shell
##if you want to use .npz inputs:
unzip ~/.local/shark_tank/<your unet>/inputs.npz

iree-benchmark-module --module_file=/path/to/output/vmfb --entry_function=forward --function_input=@arr_0.npy --function_input=1xf16 --function_input=@arr_2.npy --function_input=@arr_3.npy --function_input=@arr_4.npy  
```

## Using other supported Stable Diffusion variants with SHARK:

Currently we support the following fine-tuned versions of Stable Diffusion:
- [AnythingV3](https://huggingface.co/Linaqruf/anything-v3.0)
- [Analog Diffusion](https://huggingface.co/wavymulder/Analog-Diffusion)

use the flag `--hf_model_id=` to specify the repo-id of the model to be used.

```shell
python .\shark\examples\shark_inference\stable_diffusion\main.py --hf_model_id="Linaqruf/anything-v3.0" --max_length=77 --prompt="1girl, brown hair, green eyes, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden"
```

## Using `ckpt_loc` argument to run a custom model using a `.ckpt` file:
* To try this feature you may download a [.ckpt](https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned-fp32.ckpt) file in case you don't have a locally generated `.ckpt` file for StableDiffusion.
* Now pass the above `.ckpt` file to `ckpt_loc` command-line argument using the following :-
```shell
python3.10 main.py --precision=fp16 --device=vulkan --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd" --max_length=64 --import_mlir --ckpt_loc="/path/to/.ckpt/file" --hf_model_id="<HuggingFace repo-id>"
```
* We use a combination of 3 flags to make this feature work : `import_mlir`, `ckpt_loc` and `hf_model_id`, of which `import_mlir` needs to be present. In case `ckpt_loc` is not specified then a [default](https://huggingface.co/CompVis/stable-diffusion-v1-4) HuggingFace repo-id is run via `hf_model_id`. So, you need to specify which base model's `.ckpt` you are using via `hf_model_id`.
* Use custom model `.ckpt` files from [HuggingFace-StableDiffusion](https://huggingface.co/models?other=stable-diffusion) to generate images. And in case you want to use any variants from HuggingFace then add the mapping of the variant to their base model in [variants.json](https://github.com/nod-ai/SHARK/blob/main/shark/examples/shark_inference/stable_diffusion/resources/variants.json).
