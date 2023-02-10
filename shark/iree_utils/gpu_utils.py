# Copyright 2020 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# All the iree_gpu related functionalities go here.

import iree.runtime as ireert
import ctypes
from shark.parser import shark_args


# Get the default gpu args given the architecture.
def get_iree_gpu_args():
    ireert.flags.FUNCTION_INPUT_VALIDATION = False
    ireert.flags.parse_flags("--cuda_allow_inline_execution")
    # TODO: Give the user_interface to pass the sm_arch.
    sm_arch = get_cuda_sm_cc()
    if (
        sm_arch
        in ["sm_70", "sm_72", "sm_75", "sm_80", "sm_84", "sm_86", "sm_89"]
    ) and (shark_args.enable_tf32 == True):
        return [
            "--iree-hal-cuda-disable-loop-nounroll-wa",
            f"--iree-hal-cuda-llvm-target-arch={sm_arch}",
        ]
    else:
        return ["--iree-hal-cuda-disable-loop-nounroll-wa"]


# Get the default gpu args given the architecture.
def get_iree_rocm_args():
    ireert.flags.FUNCTION_INPUT_VALIDATION = False
    # get arch from rocminfo.
    import re
    import subprocess

    rocm_arch = re.match(
        r".*(gfx\w+)",
        subprocess.check_output(
            "rocminfo | grep -i 'gfx'", shell=True, text=True
        ),
    ).group(1)
    print(f"Found rocm arch {rocm_arch}...")
    return [
        f"--iree-rocm-target-chip={rocm_arch}",
        "--iree-rocm-link-bc=true",
        "--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode",
    ]


# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36


def get_cuda_sm_cc():
    libnames = ("libcuda.so", "libcuda.dylib", "nvcuda.dll")
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + " ".join(libnames))

    nGpus = ctypes.c_int()
    name = b" " * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print(
            "cuInit failed with error code %d: %s"
            % (result, error_str.value.decode())
        )
        return 1
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print(
            "cuDeviceGetCount failed with error code %d: %s"
            % (result, error_str.value.decode())
        )
        return 1
    print("Found %d device(s)." % nGpus.value)
    for i in range(nGpus.value):
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            print(
                "cuDeviceGet failed with error code %d: %s"
                % (result, error_str.value.decode())
            )
            return 1
        print("Device: %d" % i)
        if (
            cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
            == CUDA_SUCCESS
        ):
            print("  Name: %s" % (name.split(b"\0", 1)[0].decode()))
        if (
            cuda.cuDeviceComputeCapability(
                ctypes.byref(cc_major), ctypes.byref(cc_minor), device
            )
            == CUDA_SUCCESS
        ):
            print(
                "  Compute Capability: %d.%d"
                % (cc_major.value, cc_minor.value)
            )
    sm = f"sm_{cc_major.value}{cc_minor.value}"
    return sm
