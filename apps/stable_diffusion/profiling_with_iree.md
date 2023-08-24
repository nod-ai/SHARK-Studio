Compile / Run Instructions:

To compile .vmfb for SD (vae, unet, CLIP), run the following commands with the .mlir in your local shark_tank cache (default location for Linux users is `~/.local/shark_tank`). These will be available once the script from [this README](https://github.com/nod-ai/SHARK/blob/main/shark/examples/shark_inference/stable_diffusion/README.md) is run once.
Running the script mentioned above with the `--save_vmfb` flag will also save the .vmfb in your SHARK base directory if you want to skip straight to benchmarks.

Compile Commands FP32/FP16: 

```shell
Vulkan AMD: 
iree-compile --iree-input-type=none --iree-hal-target-backends=vulkan --iree-vulkan-target-triple=rdna2-unknown-linux /path/to/input/mlir -o /path/to/output/vmfb

#  add --mlir-print-debuginfo --mlir-print-op-on-diagnostic=true for debug
#  use –iree-input-type=auto or "mhlo_legacy" or "stablehlo" for TF models

CUDA NVIDIA:
iree-compile --iree-input-type=none --iree-hal-target-backends=cuda /path/to/input/mlir -o /path/to/output/vmfb

CPU:
iree-compile --iree-input-type=none --iree-hal-target-backends=llvm-cpu /path/to/input/mlir -o /path/to/output/vmfb
```



Run / Benchmark Command (FP32 - NCHW):
(NEED to use BS=2 since we do two forward passes to unet as a result of classifier free guidance.)

```shell
## Vulkan AMD:
iree-benchmark-module --module=/path/to/output/vmfb --function=forward --device=vulkan --input=1x4x64x64xf32 --input=1xf32 --input=2x77x768xf32 --input=f32=1.0 --input=f32=1.0

## CUDA:
iree-benchmark-module --module=/path/to/vmfb --function=forward --device=cuda  --input=1x4x64x64xf32 --input=1xf32 --input=2x77x768xf32 --input=f32=1.0 --input=f32=1.0

## CPU:
iree-benchmark-module --module=/path/to/vmfb --function=forward --device=local-task  --input=1x4x64x64xf32 --input=1xf32 --input=2x77x768xf32 --input=f32=1.0 --input=f32=1.0

```

Run via vulkan_gui for RGP Profiling:

To build the vulkan app for profiling UNet follow the instructions [here](https://github.com/nod-ai/SHARK/tree/main/cpp) and then run the following command from the cpp directory with your compiled stable_diff.vmfb
```shell
./build/vulkan_gui/iree-vulkan-gui --module=/path/to/unet.vmfb --input=1x4x64x64xf32 --input=1xf32 --input=2x77x768xf32 --input=f32=1.0 --input=f32=1.0
```

</details>
  <details>
  <summary>Debug Commands</summary>

## Debug commands and other advanced usage follows.

```shell
python txt2img.py --precision="fp32"|"fp16" --device="cpu"|"cuda"|"vulkan" --import_mlir|--no-import_mlir --prompt "enter the text" 
```

## dump all dispatch .spv and isa using amdllpc

```shell
python txt2img.py --precision="fp16" --device="vulkan" --iree-vulkan-target-triple=rdna3-unknown-linux --no-load_vmfb --dispatch_benchmarks="all" --dispatch_benchmarks_dir="SD_dispatches" --dump_isa
```

## Compile and save the .vmfb (using vulkan fp16 as an example):

```shell
python txt2img.py --precision=fp16 --device=vulkan --steps=50 --save_vmfb
```

## Capture an RGP trace

```shell
python txt2img.py --precision=fp16 --device=vulkan --steps=50 --save_vmfb --enable_rgp
```

## Run the vae module with iree-benchmark-module (NCHW, fp16, vulkan, for example):

```shell
iree-benchmark-module --module=/path/to/output/vmfb --function=forward --device=vulkan --input=1x4x64x64xf16  
```

## Run the unet module with iree-benchmark-module (same config as above):
```shell
##if you want to use .npz inputs:
unzip ~/.local/shark_tank/<your unet>/inputs.npz
iree-benchmark-module --module=/path/to/output/vmfb --function=forward --input=@arr_0.npy --input=1xf16 --input=@arr_2.npy --input=@arr_3.npy --input=@arr_4.npy  
```

</details>
