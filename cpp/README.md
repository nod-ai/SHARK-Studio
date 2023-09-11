# SHARK C/C++ Samples

These C/C++ samples can be built using CMake. The samples depend on the main
SHARK-Runtime project's C/C++ sources, including both the runtime and the compiler. 

Individual samples may require additional dependencies. Watch CMake's output
for information about which you are missing for individual samples.

On Windows we recommend using https://github.com/microsoft/vcpkg to download packages for
your system. The general setup flow looks like

*Install and activate SHARK*

```bash
source shark.venv/bin/activate #follow main repo instructions to setup your venv
```

*Install Dependencies*

```bash
vcpkg install [library] --triplet [your platform]
vcpkg integrate install

# Then pass `-DCMAKE_TOOLCHAIN_FILE=[check logs for path]` when configuring CMake
```

In Ubuntu Linux you can install

```bash
sudo apt install libsdl2-dev
```

*Build*
```bash
cd cpp
cmake -GNinja -B build/
cmake --build build/
```

*Prepare the model*
```bash
wget https://storage.googleapis.com/shark_tank/latest/resnet50_tf/resnet50_tf.mlir
iree-compile --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=vulkan --iree-llvmcpu-embedded-linker-path=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`/iree/compiler/tools/../_mlir_libs/iree-lld --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --mlir-pass-pipeline-crash-reproducer=ist/core-reproducer.mlir --iree-llvmcpu-target-cpu-features=host -iree-vulkan-target-triple=rdna2-unknown-linux  resnet50_tf.mlir -o resnet50_tf.vmfb
```
*Prepare the input*

```bash
python save_img.py
```
Note that this requires tensorflow, e.g.
```bash
python -m pip install tensorflow
```

*Run the vulkan_gui*
```bash
./build/vulkan_gui/iree-samples-resnet-vulkan-gui
```

## Other models
A tool for benchmarking other models is built and can be invoked with a command like the following
```bash
./build/vulkan_gui/iree-vulkan-gui --module-file=path/to/.vmfb --function_input=...
```
see `./build/vulkan_gui/iree-vulkan-gui --help` for an explanation on the function input. For example, stable diffusion unet can be tested with the following commands:
```bash
wget https://storage.googleapis.com/shark_tank/quinn/stable_diff_tf/stable_diff_tf.mlir
iree-compile --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=vulkan --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --iree-llvmcpu-target-cpu-features=host -iree-vulkan-target-triple=rdna2-unknown-linux  stable_diff_tf.mlir -o stable_diff_tf.vmfb
./build/vulkan_gui/iree-vulkan-gui --module-file=stable_diff_tf.vmfb --function_input=2x4x64x64xf32 --function_input=1xf32 --function_input=2x77x768xf32
```
VAE and Autoencoder are also available
```bash
# VAE
wget https://storage.googleapis.com/shark_tank/quinn/stable_diff_tf/vae_tf/vae.mlir
iree-compile --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=vulkan --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --iree-llvmcpu-target-cpu-features=host -iree-vulkan-target-triple=rdna2-unknown-linux  vae.mlir -o vae.vmfb
./build/vulkan_gui/iree-vulkan-gui --module-file=stable_diff_tf.vmfb --function_input=1x4x64x64xf32

# CLIP Autoencoder
wget https://storage.googleapis.com/shark_tank/quinn/stable_diff_tf/clip_tf/clip_autoencoder.mlir
iree-compile --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=vulkan --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false --iree-llvmcpu-target-cpu-features=host -iree-vulkan-target-triple=rdna2-unknown-linux  clip_autoencoder.mlir -o clip_autoencoder.vmfb
./build/vulkan_gui/iree-vulkan-gui --module-file=stable_diff_tf.vmfb --function_input=1x77xi32 --function_input=1x77xi32
```
