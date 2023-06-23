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

# All the iree_vulkan related functionalities go here.

from shark.iree_utils._common import run_cmd
import iree.runtime as ireert
from sys import platform
from shark.iree_utils.vulkan_target_env_utils import get_vulkan_target_env_flag


def get_metal_device_name(device_num=0):
    iree_device_dump = run_cmd("iree-run-module --dump_devices")
    iree_device_dump = iree_device_dump[0].split("\n\n")
    metal_device_list = [
        s.split("\n#")[2] for s in iree_device_dump if "--device=metal" in s
    ]
    if len(metal_device_list) == 0:
        raise ValueError("No device name found in device dump!")
    if len(metal_device_list) > 1:
        print("Following devices found:")
        for i, dname in enumerate(metal_device_list):
            print(f"{i}. {dname}")
        print(f"Choosing device: {metal_device_list[device_num]}")
    return metal_device_list[device_num]


def get_os_name():
    if platform.startswith("linux"):
        return "linux"
    elif platform == "darwin":
        return "macos"
    elif platform == "win32":
        return "windows"
    else:
        print("Cannot detect OS type, defaulting to linux.")
        return "linux"


def get_metal_target_triple(device_name):
    """This method provides a target triple str for specified vulkan device.

    Args:
        device_name (str): name of the hardware device to be used with vulkan

    Returns:
        str or None: target triple or None if no match found for given name
    """
    # Apple Targets
    if all(x in device_name for x in ("Apple", "M1")):
        triple = "m1-moltenvk-macos"
    elif all(x in device_name for x in ("Apple", "M2")):
        triple = "m1-moltenvk-macos"

    else:
        triple = None
    return triple


def get_metal_triple_flag(device_name="", device_num=0, extra_args=[]):
    for flag in extra_args:
        if "-iree-metal-target-platform=" in flag:
            print(f"Using target triple {flag.split('=')[1]}")
            return None

    if device_name == "" or device_name == [] or device_name is None:
        metal_device = get_metal_device_name(device_num=device_num)
    else:
        metal_device = device_name
    triple = get_metal_target_triple(metal_device)
    if triple is not None:
        print(
            f"Found metal device {metal_device}. Using metal target triple {triple}"
        )
        return f"-iree-metal-target-platform={triple}"
    print(
        """Optimized kernel for your target device is not added yet.
        Contact SHARK Admin on discord[https://discord.com/invite/RUqY2h2s9u]
        or pull up an issue."""
    )
    print(f"Target : {metal_device}")
    return None


def get_iree_metal_args(device_num=0, extra_args=[]):
    # res_metal_flag = ["--iree-flow-demote-i64-to-i32"]

    res_metal_flag = []
    metal_triple_flag = None
    for arg in extra_args:
        if "-iree-metal-target-platform=" in arg:
            print(f"Using target triple {arg} from command line args")
            metal_triple_flag = arg
            break

    if metal_triple_flag is None:
        metal_triple_flag = get_metal_triple_flag(
            device_num=device_num, extra_args=extra_args
        )

    if metal_triple_flag is not None:
        vulkan_target_env = get_vulkan_target_env_flag(metal_triple_flag)
        res_metal_flag.append(vulkan_target_env)
    return res_metal_flag


def set_iree_metal_runtime_flags(flags):
    for flag in flags:
        ireert.flags.parse_flags(flag)
    return
