# STABLE DIFFUSION

## Installation

Follow setup instructions in the main README.md for installing from source. Add `IMPORTER=1` before `./setup_venv.sh` or install the following after running `source shark.venv/bin/activate`:
```shell
pip install transformers
pip install diffusers
pip install scipy
```

## RUN

```shell
python main.py --precision="fp32"|"fp16" --device="cpu"|"cuda"|"vulkan" --import_mlir|--no-import_mlir --prompt "enter the text" 

```

## Compile and save the .vmfb (using vulkan fp16 as an example):

```shell
python shark/examples/shark_inference/stable_diffusion/main.py --precision=fp16 --device=vulkan --steps=50 --save_vmfb
```

## Run the module with iree-benchmark-module (NCHW, fp16, vulkan, for example):

```shell
iree-benchmark-module --module_file=/path/to/output/vmfb --entry_function=forward --device=vulkan --function_input=2x4x64x64xf16 --function_input=1xf16 --function_input=2x77x768xf16 
```
