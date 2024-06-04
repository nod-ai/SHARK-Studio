import numpy as np
import json
from random import (
    randint,
    seed as seed_random,
    getstate as random_getstate,
    setstate as random_setstate,
)

from pathlib import Path
from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
from cpuinfo import get_cpu_info


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
    "hip": "hip",
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
    "vulkan": "vulkan-spirv",
    "metal": "metal",
    "rocm": "rocm",
    "hip": "rocm",
    "intel-gpu": "opencl-spirv",
}



def get_available_devices():
    def get_devices_by_name(driver_name):

        device_list = []
        try:
            driver_name = iree_device_map(driver_name)
            device_list_dict = get_all_devices(driver_name)
            print(f"{driver_name} devices are available.")
        except:
            print(f"{driver_name} devices are not available.")
        else:
            cpu_name = get_cpu_info()["brand_raw"]
            for i, device in enumerate(device_list_dict):
                device_name = (
                    cpu_name if device["name"] == "default" else device["name"]
                )
                if "local" in driver_name:
                    device_list.append(
                        f"{device_name} => {driver_name.replace('local', 'cpu')}"
                    )
                else:
                    # for drivers with single devices
                    # let the default device be selected without any indexing
                    if len(device_list_dict) == 1:
                        device_list.append(f"{device_name} => {driver_name}")
                    else:
                        device_list.append(f"{device_name} => {driver_name}://{i}")
        return device_list

    #set_iree_runtime_flags()

    available_devices = []
    rocm_devices = get_devices_by_name("rocm")
    available_devices.extend(rocm_devices)
    # cpu_device = get_devices_by_name("cpu-sync")
    # available_devices.extend(cpu_device)
    cpu_device = get_devices_by_name("cpu-task")
    available_devices.extend(cpu_device)

    # from shark.iree_utils.vulkan_utils import (
    #     get_all_vulkan_devices,
    # )

    # vulkaninfo_list = get_all_vulkan_devices()
    # vulkan_devices = []
    # id = 0
    # for device in vulkaninfo_list:
    #     vulkan_devices.append(f"{device.strip()} => vulkan://{id}")
    #     id += 1
    # if id != 0:
    #     print(f"vulkan devices are available.")

    # available_devices.extend(vulkan_devices)
    # metal_devices = get_devices_by_name("metal")
    # available_devices.extend(metal_devices)
    # cuda_devices = get_devices_by_name("cuda")
    # available_devices.extend(cuda_devices)
    # hip_devices = get_devices_by_name("hip")
    # available_devices.extend(hip_devices)

    for idx, device_str in enumerate(available_devices):
        if "AMD Radeon(TM) Graphics =>" in device_str:
            igpu_id_candidates = [
                x.split("w/")[-1].split("=>")[0]
                for x in available_devices
                if "M Graphics" in x
            ]
            for igpu_name in igpu_id_candidates:
                if igpu_name:
                    available_devices[idx] = device_str.replace(
                        "AMD Radeon(TM) Graphics", igpu_name
                    )
                break
    return available_devices

def clean_device_info(raw_device):
    # return appropriate device and device_id for consumption by Studio pipeline
    # Multiple devices only supported for vulkan and rocm (as of now).
    # default device must be selected for all others

    device_id = None
    device = (
        raw_device
        if "=>" not in raw_device
        else raw_device.split("=>")[1].strip()
    )
    if "://" in device:
        device, device_id = device.split("://")
        if len(device_id) <= 2:
            device_id = int(device_id)

    if device not in ["hip", "rocm", "vulkan"]:
        device_id = None
    if device in ["hip", "rocm", "vulkan"] and device_id == None:
        device_id = 0
    return device, device_id

def parse_device(device_str, target_override=""):

    rt_driver, device_id = clean_device_info(device_str)
    target_backend = iree_target_map(rt_driver)
    if device_id:
        rt_device = f"{rt_driver}://{device_id}"
    else:
        rt_device = rt_driver

    if target_override:
        if "cpu" in device_str:
            rt_device = "local-task"
        return target_backend, rt_device, target_override
    match target_backend:
        case "vulkan-spirv":
            triple = get_iree_target_triple(device_str)
            return target_backend, rt_device, triple
        case "rocm":
            triple = get_rocm_target_chip(device_str)
            return target_backend, rt_device, triple
        case "llvm-cpu":
            if "Ryzen 9" in device_str:
                return target_backend, "local-task", "znver4"
            else:
                return "llvm-cpu", "local-task", "x86_64-linux-gnu"


def get_rocm_target_chip(device_str):
    # TODO: Use a data file to map device_str to target chip.
    rocm_chip_map = {
        "6700": "gfx1031",
        "6800": "gfx1030",
        "6900": "gfx1030",
        "7900": "gfx1100",
        "MI300X": "gfx942",
        "MI300A": "gfx940",
        "MI210": "gfx90a",
        "MI250": "gfx90a",
        "MI100": "gfx908",
        "MI50": "gfx906",
        "MI60": "gfx906",
        "780M": "gfx1103",
    }
    for key in rocm_chip_map:
        if key in device_str:
            return rocm_chip_map[key]
    return None


def get_all_devices(driver_name):
    """
    Inputs: driver_name
    Returns a list of all the available devices for a given driver sorted by
    the iree path names of the device as in --list_devices option in iree.
    """
    from iree.runtime import get_driver

    driver = get_driver(driver_name)
    device_list_src = driver.query_available_devices()
    device_list_src.sort(key=lambda d: d["path"])
    del driver
    return device_list_src


def get_device_mapping(driver, key_combination=3):
    """This method ensures consistent device ordering when choosing
    specific devices for execution
    Args:
        driver (str): execution driver (vulkan, cuda, rocm, etc)
        key_combination (int, optional): choice for mapping value for
            device name.
        1 : path
        2 : name
        3 : (name, path)
        Defaults to 3.
    Returns:
        dict: map to possible device names user can input mapped to desired
            combination of name/path.
    """

    driver = iree_device_map(driver)
    device_list = get_all_devices(driver)
    device_map = dict()

    def get_output_value(dev_dict):
        if key_combination == 1:
            return f"{driver}://{dev_dict['path']}"
        if key_combination == 2:
            return dev_dict["name"]
        if key_combination == 3:
            return dev_dict["name"], f"{driver}://{dev_dict['path']}"

    # mapping driver name to default device (driver://0)
    device_map[f"{driver}"] = get_output_value(device_list[0])
    for i, device in enumerate(device_list):
        # mapping with index
        device_map[f"{driver}://{i}"] = get_output_value(device)
        # mapping with full path
        device_map[f"{driver}://{device['path']}"] = get_output_value(device)
    return device_map


def get_opt_flags(model, precision="fp16"):
    iree_flags = []
    if len(cmd_opts.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={cmd_opts.iree_vulkan_target_triple}"
        )
    if "rocm" in cmd_opts.device:
        from shark.iree_utils.gpu_utils import get_iree_rocm_args

        rocm_args = get_iree_rocm_args()
        iree_flags.extend(rocm_args)
    if cmd_opts.iree_constant_folding == False:
        iree_flags.append("--iree-opt-const-expr-hoisting=False")
        iree_flags.append(
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807"
        )
    if cmd_opts.data_tiling == False:
        iree_flags.append("--iree-opt-data-tiling=False")

    if "vae" not in model:
        # Due to lack of support for multi-reduce, we always collapse reduction
        # dims before dispatch formation right now.
        iree_flags += ["--iree-flow-collapse-reduction-dims"]
    return iree_flags