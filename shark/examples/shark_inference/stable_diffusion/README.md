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

## Using custom checkpoints for a specific version of a model:
* This feature can only be used with `import_mlir` command-line argument.
* To try this feature you may download [unet-checkpoint](https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/unet/diffusion_pytorch_model.bin) and provide its path to `--unet_checkpoint` command-line argument.
```shell
python main.py --precision=fp32 --device=vulkan --import_mlir --unet_checkpoint=<path to unet's checkpoint file> --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd"
(Similarly one can use --vae_checkpoint and --clip_checkpoint command-line arguments to use custom checkpoint weights for the individual models)