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


def get_all_devices(driver_name, full_dict=False):
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
    if full_dict:
        return device_list_src
    device_list = []
    for device_dict in device_list_src:
        device_list.append(f"{driver_name}://{device_dict['path']}")
    return device_list


def _iree_device_map_vulkan(device):
    def get_selected_device_index():
        if "://" not in device:
            return 0

        _, d_index = device.split("://")
        matched_index = 0
        match_with_index = False
        if 0 <= len(d_index) <= 2:
            try:
                d_index = int(d_index)
                if d_index >= len(device_list):
                    raise IndexError()
            except ValueError:
                print(
                    f"{d_index} is not valid index or uri. Choosing device 0"
                )
                return 0
            except IndexError:
                print(
                    f"Only 0 to {len(device_list)-1} devices available and "
                    f"user requested device {d_index}. Choosing device 0"
                )
                return 0
            match_with_index = True

        # Only called when there are multiple devices
        for i, d in enumerate(device_list):
            if (match_with_index and d_index == i) or (
                not match_with_index and d == device
            ):
                matched_index = i
        return matched_index

    # only supported for vulkan as of now
    device_list = get_all_devices("vulkan")

    if len(device_list) == 1:
        print(
            f"Available vulkan device:\nvulkan://0 => {device_list[0]}\nUsing this device."
        )
        return get_device(device_list[0])

    if len(device_list) > 1:
        matched_index = get_selected_device_index(device, device_list)
        print("List of available vulkan devices:")
        for i, d in enumerate(device_list):
            print(f"vulkan://{i} => {d}")
        print(
            f"Choosing device vulkan://{matched_index}\nTo choose another device "
            f"please specify device index or uri accordingly."
        )
        return get_device(device_list[matched_index])

    print(
        f"No device found! returning device corresponding to driver name: vulkan"
    )
    return _IREE_DEVICE_MAP["vulkan"]


def iree_device_map(device):
    uri_parts = device.split("://", 2)
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
