On your host install your Nvidia or AMD gpu drivers. 

**HOST Setup**

*Ubuntu 23.04 Nvidia*
```
sudo ubuntu-drivers install
```

Install [docker](https://docs.docker.com/engine/install/ubuntu/) and the post-install to run as a [user](https://docs.docker.com/engine/install/linux-postinstall/)

Install Nvidia [Container and register it](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). In Ubuntu 23.04 systems follow [this](https://github.com/NVIDIA/nvidia-container-toolkit/issues/72#issuecomment-1584574298)


Build docker with :

```
docker build . -f Dockerfile-ubuntu-22.04 -t shark/dev-22.04:5.6 --build-arg=ROCM_VERSION=5.6 --build-arg=AMDGPU_VERSION=5.6 --build-arg=APT_PREF="Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" --build-arg=IMAGE_NAME=nvidia/cuda --build-arg=TARGETARCH=amd64
```

Run with:

*CPU*

```
docker run  -it docker.io/shark/dev-22.04:5.6
```

*Nvidia GPU*

```
docker run --rm -it --gpus all docker.io/shark/dev-22.04:5.6
```

*AMD GPUs*

```
docker run --device /dev/kfd --device /dev/dri  docker.io/shark/dev-22.04:5.6
```

More AMD instructions are [here](https://docs.amd.com/en/latest/deploy/docker.html)
