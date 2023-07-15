Build with :

```
docker build . -f Dockerfile-ubuntu-22.04 -t shark/dev-22.04:5.6 --build-arg=ROCM_VERSION=5.6 --build-arg=AMDGPU_VERSION=5.6 --build-arg=APT_PREF="Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" --build-arg=IMAGE_NAME=nvidia/cuda --build-arg=TARGETARCH=amd64
```

Run with:

*CPU*

```
docker run  -it docker.io/shark/dev-22.04:5.6
```

