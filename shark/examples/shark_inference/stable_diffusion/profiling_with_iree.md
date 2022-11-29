Compile / Run Instructions:

To compile .vmfb for SD (vae, unet, CLIP), run the following commands with the .mlir in your local shark_tank cache (default location for Linux users is `~/.local/shark_tank`). These will be available once the script from [this README](https://github.com/nod-ai/SHARK/blob/main/shark/examples/shark_inference/stable_diffusion/README.md) is run once.
Running the script mentioned above with the `--save_vmfb` flag will also save the .vmfb in your SHARK base directory if you want to skip straight to benchmarks.

Compile Commands FP32/FP16: 

```shell
Vulkan AMD: 
iree-compile --iree-input-type=none --iree-hal-target-backends=vulkan --iree-vulkan-target-triple=rdna2-unknown-linux --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 /path/to/input/mlir -o /path/to/output/vmfb

#  add --mlir-print-debuginfo --mlir-print-op-on-diagnostic=true for debug
#  use –iree-input-type=mhlo for tf models

CUDA NVIDIA:
iree-compile --iree-input-type=none --iree-hal-target-backends=cuda --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 /path/to/input/mlir -o /path/to/output/vmfb

CPU:
iree-compile --iree-input-type=none --iree-hal-target-backends=llvm-cpu  --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 /path/to/input/mlir -o /path/to/output/vmfb
```



Run / Benchmark Command (FP32 - NCHW):
(NEED to use BS=2 since we do two forward passes to unet as a result of classifier free guidance.)

```shell
## Vulkan AMD:
iree-benchmark-module --module_file=/path/to/output/vmfb --entry_function=forward --device=vulkan --function_input=1x4x64x64xf32 --function_input=1xf32 --function_input=2x77x768xf32 --function_input=f32=1.0 --function_input=f32=1.0

## CUDA:
iree-benchmark-module --module_file=/path/to/vmfb --entry_function=forward --device=cuda  --function_input=1x4x64x64xf32 --function_input=1xf32 --function_input=2x77x768xf32 --function_input=f32=1.0 --function_input=f32=1.0

## CPU:
iree-benchmark-module --module_file=/path/to/vmfb --entry_function=forward --device=local-task  --function_input=1x4x64x64xf32 --function_input=1xf32 --function_input=2x77x768xf32 --function_input=f32=1.0 --function_input=f32=1.0

```

Run via vulkan_gui for RGP Profiling:

To build the vulkan app for profiling UNet follow the instructions [here](https://github.com/nod-ai/SHARK/tree/main/cpp) and then run the following command from the cpp directory with your compiled stable_diff.vmfb
```shell
./build/vulkan_gui/iree-vulkan-gui --module_file=/path/to/unet.vmfb --function_input=1x4x64x64xf32 --function_input=1xf32 --function_input=2x77x768xf32 --function_input=f32=1.0 --function_input=f32=1.0
```
