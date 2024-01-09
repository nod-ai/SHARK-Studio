import numpy as np
import json
from random import (
    randint,
    seed as seed_random,
    getstate as random_getstate,
    setstate as random_setstate,
)

from pathlib import Path
#from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
from cpuinfo import get_cpu_info

# TODO: migrate these utils to studio
from shark.iree_utils.vulkan_utils import (
    set_iree_vulkan_runtime_flags,
    get_vulkan_target_triple,
    get_iree_vulkan_runtime_flags,
)


def get_available_devices():
    def get_devices_by_name(driver_name):
        from shark.iree_utils._common import iree_device_map

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

    set_iree_runtime_flags()

    available_devices = []
    from shark.iree_utils.vulkan_utils import (
        get_all_vulkan_devices,
    )

    vulkaninfo_list = get_all_vulkan_devices()
    vulkan_devices = []
    id = 0
    for device in vulkaninfo_list:
        vulkan_devices.append(f"{device.strip()} => vulkan://{id}")
        id += 1
    if id != 0:
        print(f"vulkan devices are available.")
    available_devices.extend(vulkan_devices)
    metal_devices = get_devices_by_name("metal")
    available_devices.extend(metal_devices)
    cuda_devices = get_devices_by_name("cuda")
    available_devices.extend(cuda_devices)
    rocm_devices = get_devices_by_name("rocm")
    available_devices.extend(rocm_devices)
    cpu_device = get_devices_by_name("cpu-sync")
    available_devices.extend(cpu_device)
    cpu_device = get_devices_by_name("cpu-task")
    available_devices.extend(cpu_device)
    return available_devices

def set_iree_runtime_flags():
    # TODO: This function should be device-agnostic and piped properly
    # to general runtime driver init.
    vulkan_runtime_flags = get_iree_vulkan_runtime_flags()

    set_iree_vulkan_runtime_flags(flags=vulkan_runtime_flags)


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
    from shark.iree_utils._common import iree_device_map

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


def map_device_to_name_path(device, key_combination=3):
    """Gives the appropriate device data (supported name/path) for user
        selected execution device
    Args:
        device (str): user
        key_combination (int, optional): choice for mapping value for
            device name.
        1 : path
        2 : name
        3 : (name, path)
        Defaults to 3.
    Raises:
        ValueError:
    Returns:
        str / tuple: returns the mapping str or tuple of mapping str for
        the device depending on key_combination value
    """
    driver = device.split("://")[0]
    device_map = get_device_mapping(driver, key_combination)
    try:
        device_mapping = device_map[device]
    except KeyError:
        raise ValueError(f"Device '{device}' is not a valid device.")
    return device_mapping


# Generate and return a new seed if the provided one is not in the
# supported range (including -1)
def sanitize_seed(seed: int | str):
    seed = int(seed)
    uint32_info = np.iinfo(np.uint32)
    uint32_min, uint32_max = uint32_info.min, uint32_info.max
    if seed < uint32_min or seed >= uint32_max:
        seed = randint(uint32_min, uint32_max)
    return seed


# take a seed expression in an input format and convert it to
# a list of integers, where possible
def parse_seed_input(seed_input: str | list | int):
    if isinstance(seed_input, str):
        try:
            seed_input = json.loads(seed_input)
        except (ValueError, TypeError):
            seed_input = None

    if isinstance(seed_input, int):
        return [seed_input]

    if isinstance(seed_input, list) and all(type(seed) is int for seed in seed_input):
        return seed_input

    raise TypeError(
        "Seed input must be an integer or an array of integers in JSON format"
    )