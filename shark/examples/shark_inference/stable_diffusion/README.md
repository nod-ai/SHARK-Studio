# STABLE DIFFUSION

## Installation

Follow setup instructions in the main [README.md](https://github.com/nod-ai/SHARK#readme) for regular usage. 

 
## Using other supported Stable Diffusion variants with SHARK:

Currently we support fine-tuned versions of Stable Diffusion such as:
- [AnythingV3](https://huggingface.co/Linaqruf/anything-v3.0)
- [Analog Diffusion](https://huggingface.co/wavymulder/Analog-Diffusion)

use the flag `--hf_model_id=` to specify the repo-id of the model to be used.

```shell
python .\shark\examples\shark_inference\stable_diffusion\main.py --hf_model_id="Linaqruf/anything-v3.0" --max_length=77 --prompt="1girl, brown hair, green eyes, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden" --no-use_tuned
```

## Run a custom model using a `.ckpt` / `.safetensors` checkpoint file:
* Ensure you don't have any `.yaml` file at the root directory of SHARK - best would be to ensure you're on the latest `main` branch and use `--clear_all` the first time you're running the command for inference.
* Install `pytorch_lightning` by running :-
```shell
pip install pytorch_lightning
```
NOTE: This is needed to process [ckpt file of runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt).
* Download a [.ckpt](https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.0-pruned-fp32.ckpt) file in case you don't have a locally generated `.ckpt` file for StableDiffusion.

* Now pass the above `.ckpt` file to `ckpt_loc` command-line argument using the following :-
```shell
python3.10 main.py --precision=fp16 --device=vulkan --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd" --max_length=64 --import_mlir --ckpt_loc="/path/to/.ckpt/file" --no-use_tuned
```
* We use a combination of 2 flags to make this feature work : `import_mlir` and `ckpt_loc`.
* In case `ckpt_loc` is NOT specified then a [default](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) HuggingFace repo-id is run via `hf_model_id`. So, two ways to use `import_mlir` :-
- With `hf_model_id` to run HuggingFace's StableDiffusion variants.
- With `ckpt_loc` to run a StableDiffusion variant with a `.ckpt` or `.safetensors` checkpoint file

* Use custom model `.ckpt` files from [HuggingFace-StableDiffusion](https://huggingface.co/models?other=stable-diffusion) to generate images.
* You may also try out [.safetensors file of Protogen x3.4 of civitai.com](https://civitai.com/models/3666/protogen-x34-photorealism-official-release) and provide the `.safetensors` path to `ckpt_loc` flag.
* NOTE: Ensure that the `.ckpt` or `.safetensors` file are part of the path passed to `ckpt_loc` flag. Eg: `--ckpt_loc="/path/to/checkpoint/file/name_of_checkpoint.ckpt` OR `--ckpt_loc="/path/to/checkpoint/file/name_of_checkpoint.safetensors`. Also ensure that you're using `--no-use_tuned` flag in your run command.


## Running the model for a `batch_size` and for a set of `runs`:
We currently support batch size in the range `[1, 3]`.
You can specify batch size using `batch_size` flag (defaults to `1`) and the number of times you want to run the model using `runs` flag (defaults to `1`).
In total, you'll be able to generate `batch_size * runs` number of images.
- Usage 1: Using the same prompt -
```shell
python3.10 main.py --precision=fp16 --device=vulkan --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd" --max_length=64 --import_mlir --hf_model_id="runwayml/stable-diffusion-v1-5" --batch_size=3 --no-use_tuned
```
The example above generates `3` different images in total with the same prompt `tajmahal, oil on canvas, sunflowers, 4k, uhd`.
- Usage 2: Using different prompts -
```shell
python3.10 main.py --precision=fp16 --device=vulkan --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd" --max_length=64 --import_mlir --hf_model_id="runwayml/stable-diffusion-v1-5" --batch_size=3 -p="batman riding a horse, oil on canvas, 4k, uhd" -p="superman riding a horse, oil on canvas, 4k, uhd" --no-use_tuned
```
The example above generates `1` image for each different prompt, thus generating `3` images in total.
- Usage 3: Using `runs` -
```shell
python3.10 main.py --precision=fp16 --device=vulkan --prompt="tajmahal, oil on canvas, sunflowers, 4k, uhd" --max_length=64 --import_mlir --hf_model_id="runwayml/stable-diffusion-v1-5" --batch_size=2 --runs=3 --no-use_tuned
```
The example above generates `6` different images in total, `2` images for each `runs`.

</details>
  <details>
  <summary>Debug Commands</summary>

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

</details>
