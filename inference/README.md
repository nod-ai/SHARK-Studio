# dSHARK Backend

The triton backend for shark.

# Build

Install dSHARK

```
git clone https://github.com/NodLabs/dSHARK.git
# skip above step if dshark is already installed
cd inference
```

install dependancies

```
$ apt-get install patchelf rapidjson-dev python3-dev
git submodule update --init
```

update the submodules of iree

```
cd thirdparty/iree

git submodule update --init
```

Next, make the backend and install it

```
cd ../..
mkdir build && cd build
cmake -DTRITON_ENABLE_GPU=ON \
-DIREE_HAL_DRIVER_CUDA=ON \
-DIREE_TARGET_BACKEND_CUDA=ON \
-DMLIR_ENABLE_CUDA_RUNNER=ON \
-DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
-DTRITON_BACKEND_REPO_TAG=r22.02 \
-DTRITON_CORE_REPO_TAG=r22.02 \
-DTRITON_COMMON_REPO_TAG=r22.02 ..

make install
```

There should be a file at /build/install/backends/dshark/libtriton_dshark.so.  Copy it into your triton server image.  https://github.com/triton-inference-server/server/blob/main/docs/compose.md#triton-with-unsupported-and-custom-backends




