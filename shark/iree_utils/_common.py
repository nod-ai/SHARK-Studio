# Copyright 2023 The Nod Team. All rights reserved.
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

## Common utilities to be shared by iree utilities.
import functools
import os
import sys
import subprocess


def run_cmd(cmd, debug=False, raise_err=False):
    """
    Inputs:
      cmd : cli command string.
      debug : if True, prints debug info
      raise_err : if True, raise exception to caller
    """
    if debug:
        print("IREE run command: \n\n")
        print(cmd)
        print("\n\n")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        stdout = result.stdout.decode()
        stderr = result.stderr.decode()
        return stdout, stderr
    except subprocess.CalledProcessError as e:
        if raise_err:
            raise Exception from e
        else:
            print(e.output)
            sys.exit(f"Exiting program due to error running {cmd}")


def iree_device_map(device):
    uri_parts = device.split("://", 2)
    iree_driver = (
        _IREE_DEVICE_MAP[uri_parts[0]]
        if uri_parts[0] in _IREE_DEVICE_MAP
        else uri_parts[0]
    )
    if len(uri_parts) == 1:
        return iree_driver
    elif "rocm" in uri_parts:
        return "rocm"
    else:
        return f"{iree_driver}://{uri_parts[1]}"


def get_supported_device_list():
    return list(_IREE_DEVICE_MAP.keys())


_IREE_DEVICE_MAP = {
    "cpu": "local-task",
    "cpu-task": "local-task",
    "cpu-sync": "local-sync",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "metal",
    "rocm": "rocm",
    "intel-gpu": "level_zero",
}


def iree_target_map(device):
    if "://" in device:
        device = device.split("://")[0]
    return _IREE_TARGET_MAP[device] if device in _IREE_TARGET_MAP else device


_IREE_TARGET_MAP = {
    "cpu": "llvm-cpu",
    "cpu-task": "llvm-cpu",
    "cpu-sync": "llvm-cpu",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "metal",
    "rocm": "rocm",
    "intel-gpu": "opencl-spirv",
}


# Finds whether the required drivers are installed for the given device.
@functools.cache
def check_device_drivers(device):
    """
    Checks necessary drivers present for gpu and vulkan devices
    False => drivers present!
    """
    if "://" in device:
        device = device.split("://")[0]

    from iree.runtime import get_driver

    device_mapped = iree_device_map(device)

    try:
        _ = get_driver(device_mapped)
    except ValueError as ve:
        print(
            f"[ERR] device `{device}` not registered with IREE. "
            "Ensure IREE is configured for use with this device.\n"
            f"Full Error: \n {repr(ve)}"
        )
        return True
    except RuntimeError as re:
        print(
            f"[ERR] Failed to get driver for {device} with error:\n{repr(re)}"
        )
        return True

    # Unknown device. We assume drivers are installed.
    return False


# Installation info for the missing device drivers.
def device_driver_info(device):
    device_driver_err_map = {
        "cuda": {
            "debug": "Try `nvidia-smi` on system to check.",
            "solution": " from https://www.nvidia.in/Download/index.aspx?lang=en-in for your system.",
        },
        "vulkan": {
            "debug": "Try `vulkaninfo` on system to check.",
            "solution": " from https://vulkan.lunarg.com/sdk/home for your distribution.",
        },
        "metal": {
            "debug": "Check if Bare metal is supported and enabled on your system.",
            "solution": ".",
        },
        "rocm": {
            "debug": f"Try `{'hip' if sys.platform == 'win32' else 'rocm'}info` on system to check.",
            "solution": " from https://rocm.docs.amd.com/en/latest/rocm.html for your system.",
        },
    }

    if device in device_driver_err_map:
        err_msg = (
            f"Required drivers for {device} not found. {device_driver_err_map[device]['debug']} "
            f"Please install the required drivers{device_driver_err_map[device]['solution']} "
            f"For further assistance please reach out to the community on discord [https://discord.com/invite/RUqY2h2s9u]"
            f" and/or file a bug at https://github.com/nod-ai/SHARK/issues"
        )
        return err_msg
    else:
        return f"{device} is not supported."
