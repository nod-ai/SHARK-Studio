# IREE C/C++ Samples

These C/C++ samples can be built using CMake. The samples depend on the main
IREE project's C/C++ sources, including both the runtime and the compiler. See
https://iree-org.github.io/iree/building-from-source/getting-started/ for general
information about build system installation and options, then from this project
run

```bash
cmake -GNinja -B ./build/ -S .
cmake --build ./build/
```

Individual samples may require additional dependencies. Watch CMake's output
for information about which you are missing for individual samples.

We recommend using https://github.com/microsoft/vcpkg to download packages for
your system. The general setup flow looks like

```bash
vcpkg install [library] --triplet [your platform]
vcpkg integrate install

# Then pass `-DCMAKE_TOOLCHAIN_FILE=[check logs for path]` when configuring CMake
```
