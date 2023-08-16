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


def run_cmd(cmd, debug=False):
    """
    Inputs: cli command string.
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
    """Checks necessary drivers present for gpu and vulkan devices"""
    if "://" in device:
        device = device.split("://")[0]

    if device == "cuda":
        try:
            subprocess.check_output("nvidia-smi")
        except Exception:
            return True
    elif device in ["vulkan"]:
        try:
            subprocess.check_output("vulkaninfo")
        except Exception:
            return True
    elif device == "metal":
        return False
    elif device in ["intel-gpu"]:
        try:
            subprocess.check_output(["dpkg", "-L", "intel-level-zero-gpu"])
            return False
        except Exception:
            return True
    elif device == "cpu":
        return False
    elif device == "rocm":
        try:
            if sys.platform == "win32":
                subprocess.check_output("hipinfo")
            else:
                subprocess.check_output("rocminfo")
        except Exception:
            return True

    # Unknown device. We assume drivers are installed.
    return False


# Installation info for the missing device drivers.
def device_driver_info(device):
    if device == "cuda":
        return "nvidia-smi not found, please install the required drivers from https://www.nvidia.in/Download/index.aspx?lang=en-in"
    elif device in ["metal", "vulkan"]:
        return "vulkaninfo not found, Install from https://vulkan.lunarg.com/sdk/home or your distribution"
    elif device == "rocm":
        return "rocm info not found. Please install rocm"
    else:
        return f"{device} is not supported."
