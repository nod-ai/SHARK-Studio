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
