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
from iree.runtime import get_driver, get_device


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


def get_all_devices(driver_name):
    """
    Inputs: driver_name
    Returns a list of all the available devices for a given driver sorted by
    the iree path names of the device as in --list_devices option in iree.
    Set `full_dict` flag to True to get a dict
    with `path`, `name` and `device_id` for all devices
    """
    driver = get_driver(driver_name)
    device_list_src = driver.query_available_devices()
    device_list_src.sort(key=lambda d: d["path"])
    return device_list_src


def create_map_device_to_key(driver, key):
    # key can only be path, name, device id
    device_list = get_all_devices(driver)
    device_map = dict()
    # mapping driver name to default device (driver://0)
    device_map[f"{driver}"] = f"{device_list[0][key]}"
    for i, device in enumerate(device_list):
        # mapping with index
        device_map[f"{driver}://{i}"] = f"{device[key]}"
        # mapping with full path
        device_map[f"{driver}://{device['path']}"] = f"{device[key]}"

    return device_map


def map_device_to_path(device):
    driver = device.split("://")[0]
    device_map = create_map_device_to_key(driver, "path")
    try:
        device_path = device_map[device]
    except KeyError:
        raise Exception(f"Device {device} is not a valid device.")
    return f"{driver}://{device_path}"


def map_device_to_name(device):
    driver = device.split("://")[0]
    device_map = create_map_device_to_key(driver, "name")
    try:
        device_name = device_map[device]
    except KeyError:
        raise Exception(f"Device {device} is not a valid device.")
    return device_name


def iree_device_map(device):
    uri_parts = device.split("://", 1)
    if len(uri_parts) == 1:
        return _IREE_DEVICE_MAP[uri_parts[0]]
    else:
        return f"{_IREE_DEVICE_MAP[uri_parts[0]]}://{uri_parts[1]}"


def get_supported_device_list():
    return list(_IREE_DEVICE_MAP.keys())


_IREE_DEVICE_MAP = {
    "cpu": "local-task",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "vulkan",
    "rocm": "rocm",
    "intel-gpu": "level_zero",
}


def iree_target_map(device):
    if "://" in device:
        device = device.split("://")[0]
    return _IREE_TARGET_MAP[device]


_IREE_TARGET_MAP = {
    "cpu": "llvm-cpu",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "vulkan",
    "rocm": "rocm",
    "intel-gpu": "opencl-spirv",
}


# Finds whether the required drivers are installed for the given device.
def check_device_drivers(device):
    """Checks necessary drivers present for gpu and vulkan devices"""
    if "://" in device:
        device = device.split("://")[0]

    if device == "cuda":
        try:
            subprocess.check_output("nvidia-smi")
        except Exception:
            return True
    elif device in ["metal", "vulkan"]:
        try:
            subprocess.check_output("vulkaninfo")
        except Exception:
            return True
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
            subprocess.check_output("rocminfo")
        except Exception:
            return True
    # Unknown device.
    else:
        return True

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
