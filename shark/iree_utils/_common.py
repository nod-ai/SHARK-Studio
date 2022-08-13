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

## Common utilities to be shared by iree utilities.

import os
import sys
import subprocess


def run_cmd(cmd):
    """
    Inputs: cli command string.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        result_str = result.stdout.decode()
        return result_str
    except Exception:
        sys.exit("Exiting program due to error running:", cmd)


IREE_DEVICE_MAP = {
    "cpu": "local-task",
    "gpu": "cuda",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "vulkan",
    "rocm": "rocm",
}

IREE_TARGET_MAP = {
    "cpu": "llvm-cpu",
    "gpu": "cuda",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "vulkan",
    "rocm": "rocm",
}

# Finds whether the required drivers are installed for the given device.
def check_device_drivers(device):
    """Checks necessary drivers present for gpu and vulkan devices"""
    if device in ["gpu", "cuda"]:
        try:
            subprocess.check_output("nvidia-smi")
        except Exception:
            return True
    elif device in ["metal", "vulkan"]:
        try:
            subprocess.check_output("vulkaninfo")
        except Exception:
            return True
    elif device == "cpu":
        return False
    # Unknown device.
    else:
        return True

    return False


# Installation info for the missing device drivers.
def device_driver_info(device):
    if device in ["gpu", "cuda"]:
        return "nvidia-smi not found, please install the required drivers from https://www.nvidia.in/Download/index.aspx?lang=en-in"
    elif device in ["metal", "vulkan"]:
        return "vulkaninfo not found, Install from https://vulkan.lunarg.com/sdk/home or your distribution"
    else:
        return f"{device} is not supported."
