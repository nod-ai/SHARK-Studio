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

use the flag `--variant=` to specify the model to be used.

```shell
python .\shark\examples\shark_inference\stable_diffusion\main.py --variant=anythingv3 --max_length=77 --prompt="1girl, brown hair, green eyes, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden"
```

## Using `custom_model` argument to run a custom model:
* To try this feature you may download a [.ckpt](https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned-fp32.ckpt) file in case you don't have a locally generated `.ckpt` file for StableDiffusion.
* Now pass the `.ckpt` file to [convert_original_stable_diffusion_to_diffusers.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py). Eg:
```shell
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path="/path/to/ckpt/file" --dump_path="/path/to/dump/the/diffusers/structure"
```
This would end up segregating the `.ckpt` file to match the structure of `diffusers`.
* Now pass the diffusers' structure generated above to `custom_model` command-line argument using the following :-
```shell
python3.10 main.py --precision=fp32 --device=cuda --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd" --max_length=77 --version="v1_4" --custom_model="/path/to/dump/the/diffusers/structure"
```