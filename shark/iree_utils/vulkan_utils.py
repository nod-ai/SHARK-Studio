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

# All the iree_vulkan related functionalities go here.

from os import linesep
from shark.iree_utils._common import run_cmd, get_all_devices
import iree.runtime as ireert
from sys import platform


def get_vulkan_device_name(device=None):

    vulkaninfo_dump = run_cmd("vulkaninfo").split(linesep)
    device_name_trimmed = lambda s: s.strip().split("=")[1].strip()
    vulkaninfo_list = [
        device_name_trimmed(s) for s in vulkaninfo_dump if "deviceName" in s
    ]
    if len(vulkaninfo_list) == 0:
        raise ValueError("No device name found in VulkanInfo!")

    devices_list = get_all_devices("vulkan", full_dict=True)

    def get_device_index():

        if "://" not in device:
            return 0
        if len(devices_list) == 1:
            return 0

        _, device_path = device.split("://")

        if 0 <= len(device_path) <= 2:
            try:
                device_index = int(device_path)
                _ = devices_list[device_index]["name"]
            except ValueError:
                print(
                    f"{device_path} is not valid index or uri. Choosing device 0"
                )
                return 0
            except IndexError:
                print(
                    f"Only 0 to {len(devices_list)-1} devices available and "
                    f"user requested device {device_path}. Choosing device 0"
                )
                return 0
            return device_index

        for i, d in enumerate(devices_list):
            if d["path"] == device_path:
                return i
        return 0

    device_name = devices_list[get_device_index()]["name"]
    # verify device name is found in vulkan info
    # TODO : better way could be finding closest match in vulkan info
    if device_name not in vulkaninfo_list:
        print(f"{device_name} not found in devices listed by vulkan info")
        print(f"List of devices from Vulkan info:")
        for i, d in enumerate(vulkaninfo_list):
            print(f"{i}. {d}")
        print(
            f"Choosing device 0. {vulkaninfo_list[0]} for vulkan target triple identification."
        )
        device_name = vulkaninfo_list[0]

    return device_name


<<<<<<< HEAD
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


def get_vulkan_triple_flag(extra_args=[]):
    if "-iree-vulkan-target-triple=" in " ".join(extra_args):
        print(f"Using target triple from command line args")
        return None
    system_os = get_os_name()
    vulkan_device = get_vulkan_device_name()
=======
def get_vulkan_triple_flag(device=None, extra_args=[]):
    if "-iree-vulkan-target-triple=" in " ".join(extra_args):
        print(f"Using target triple from command line args")
        return None

    vulkan_device = get_vulkan_device_name(device=device)
>>>>>>> 768b23a (Get appropriate target triple for specified vulkan device)
    if all(x in vulkan_device for x in ("Apple", "M1")):
        print(f"Found {vulkan_device} Device. Using m1-moltenvk-macos")
        return "-iree-vulkan-target-triple=m1-moltenvk-macos"
    elif all(x in vulkan_device for x in ("Apple", "M2")):
        print("Found Apple M2 Device. Using m1-moltenvk-macos")
        return "-iree-vulkan-target-triple=m1-moltenvk-macos"
    elif all(x in vulkan_device for x in ("A100", "SXM4")):
        print(
            f"Found {vulkan_device} Device. Using ampere-rtx3080-{system_os}"
        )
        return f"-iree-vulkan-target-triple=ampere-rtx3080-{system_os}"
    elif all(x in vulkan_device for x in ("RTX", "3090")):
        print(
            f"Found {vulkan_device} Device. Using ampere-rtx3090-{system_os}"
        )
        return f"-iree-vulkan-target-triple=ampere-rtx3090-{system_os}"
    elif all(x in vulkan_device for x in ("RTX", "4090")):
        print(
            f"Found {vulkan_device} Device. Using ampere-rtx3090-{system_os}"
        )
        return f"-iree-vulkan-target-triple=ampere-rtx3090-{system_os}"
    elif all(x in vulkan_device for x in ("AMD", "7900")):
        print(f"Found {vulkan_device} Device. Using rdna3-7900-{system_os}")
        return f"-iree-vulkan-target-triple=rdna3-7900-{system_os}"
    elif any(x in vulkan_device for x in ("AMD", "Radeon")):
        print(f"Found AMD device. Using rdna2-unknown-{system_os}")
        return f"-iree-vulkan-target-triple=rdna2-unknown-{system_os}"
    else:
        print(
            """Optimized kernel for your target device is not added yet.
            Contact SHARK Admin on discord[https://discord.com/invite/RUqY2h2s9u]
            or pull up an issue."""
        )
        print(f"Target : {vulkan_device}")
        return None


def get_iree_vulkan_args(device=None, extra_args=[]):
    # vulkan_flag = ["--iree-flow-demote-i64-to-i32"]
    vulkan_flag = []
    vulkan_triple_flag = get_vulkan_triple_flag(
        device=device, extra_args=extra_args
    )
    if vulkan_triple_flag is not None:
        vulkan_flag.append(vulkan_triple_flag)
    return vulkan_flag


def set_iree_vulkan_runtime_flags(flags):
    for flag in flags:
        ireert.flags.parse_flags(flag)
    return
