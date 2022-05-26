# dSHARK Backend

The triton backend for shark.

# Build

Install dSHARK

```
git clone https://github.com/nod-ai/SHARK.git
# skip above step if dshark is already installed
cd dSHARK/inference
```

install dependancies

```
apt-get install patchelf rapidjson-dev python3-dev
git submodule update --init
```

update the submodules of iree

```
cd thirdparty/shark-runtime
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

# Incorporating into Triton

There are much more in depth explenations for the following steps in triton's documentation:
https://github.com/triton-inference-server/server/blob/main/docs/compose.md#triton-with-unsupported-and-custom-backends

There should be a file at /build/install/backends/dshark/libtriton_dshark.so.  You will need to copy it into your triton server image.  
More documentation is in the link above, but to create the docker image, you need to run the compose.py command in the triton-backend server repo


To first build your image, clone the tritonserver repo.

```
git clone https://github.com/triton-inference-server/server.git
```

then run `compose.py` to build a docker compose file 
```
cd server
python3 compose.py --repoagent checksum --dry-run
```

Because dshark is a third party backend, you will need to manually modify the `Dockerfile.compose` to include the dshark backend.  To do this, in the Dockerfile.compose file produced, copy this line.
the dshark backend will be located in the build folder from earlier under `/build/install/backends`

```
COPY /path/to/build/install/backends/dshark /opt/tritonserver/backends/dshark
```

Next run 
```
docker build -t tritonserver_custom -f Dockerfile.compose .
docker run -it --gpus=1 --net=host -v/path/to/model_repos:/models  tritonserver_custom:latest tritonserver --model-repository=/models
```

where `path/to/model_repos` is where you are storing the models you want to run

if your not using gpus, omit `--gpus=1`

```
docker run -it  --net=host -v/path/to/model_repos:/models  tritonserver_custom:latest tritonserver --model-repository=/models
```

# Setting up a model

to include a model in your backend, add a directory with your model name to your model repository directory.  examples of models can be seen here: https://github.com/triton-inference-server/backend/tree/main/examples/model_repos/minimal_models

make sure to adjust the input correctly in the config.pbtxt file, and save a vmfb file under 1/model.vmfb

# CUDA

if you're having issues with cuda, make sure your correct drivers are installed, and that `nvidia-smi` works, and also make sure that the nvcc compiler is on the path.





