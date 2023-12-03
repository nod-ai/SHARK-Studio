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

import functools
import iree.runtime as ireert
import ctypes
import sys
from subprocess import CalledProcessError
from shark.parser import shark_args
from shark.iree_utils._common import run_cmd

# TODO: refactor to rocm and cuda utils


# Get the default gpu args given the architecture.
@functools.cache
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
            f"--iree-hal-cuda-llvm-target-arch={sm_arch}",
        ]
    else:
        return []


def check_rocm_device_arch_in_args(extra_args):
    # Check if the target arch flag for rocm device present in extra_args
    for flag in extra_args:
        if "iree-rocm-target-chip" in flag:
            flag_arch = flag.split("=")[1]
            return flag_arch
    return None


def get_rocm_device_arch(device_num=0, extra_args=[]):
    # ROCM Device Arch selection:
    # 1 : User given device arch using `--iree-rocm-target-chip` flag
    # 2 : Device arch from `iree-run-module --dump_devices=rocm` for device on index <device_num>
    # 3 : default arch : gfx1100

    arch_in_flag = check_rocm_device_arch_in_args(extra_args)
    if arch_in_flag is not None:
        print(
            f"User Specified rocm target device arch from flag : {arch_in_flag} will be used"
        )
        return arch_in_flag

    arch_in_device_dump = None

    # get rocm arch from iree dump devices
    def get_devices_info_from_dump(dump):
        from os import linesep

        dump_clean = list(
            filter(
                lambda s: "--device=rocm" in s or "gpu-arch-name:" in s,
                dump.split(linesep),
            )
        )
        arch_pairs = [
            (
                dump_clean[i].split("=")[1].strip(),
                dump_clean[i + 1].split(":")[1].strip(),
            )
            for i in range(0, len(dump_clean), 2)
        ]
        return arch_pairs

    dump_device_info = None
    try:
        dump_device_info = run_cmd(
            "iree-run-module --dump_devices=rocm", raise_err=True
        )
    except Exception as e:
        print("could not execute `iree-run-module --dump_devices=rocm`")

    if dump_device_info is not None:
        device_num = 0 if device_num is None else device_num
        device_arch_pairs = get_devices_info_from_dump(dump_device_info[0])
        if len(device_arch_pairs) > device_num:  # can find arch in the list
            arch_in_device_dump = device_arch_pairs[device_num][1]

    if arch_in_device_dump is not None:
        print(f"Found ROCm device arch : {arch_in_device_dump}")
        return arch_in_device_dump

    default_rocm_arch = "gfx1100"
    print(
        "Did not find ROCm architecture from `--iree-rocm-target-chip` flag"
        "\n or from `iree-run-module --dump_devices=rocm` command."
        f"\nUsing {default_rocm_arch} as ROCm arch for compilation."
    )
    return default_rocm_arch


# Get the default gpu args given the architecture.
def get_iree_rocm_args(device_num=0, extra_args=[]):
    ireert.flags.FUNCTION_INPUT_VALIDATION = False
    rocm_flags = ["--iree-rocm-link-bc=true"]

    if check_rocm_device_arch_in_args(extra_args) is None:
        rocm_arch = get_rocm_device_arch(device_num, extra_args)
        rocm_flags.append(f"--iree-rocm-target-chip={rocm_arch}")

    return rocm_flags


# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36


@functools.cache
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
