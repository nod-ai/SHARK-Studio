import os
import gc
from pathlib import Path
import numpy as np
from random import randint
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from shark.iree_utils.vulkan_utils import (
    set_iree_vulkan_runtime_flags,
    get_vulkan_target_triple,
)
from shark.iree_utils.gpu_utils import get_cuda_sm_cc
from apps.stable_diffusion.src.utils.stable_args import args
from apps.stable_diffusion.src.utils.resources import opt_flags
from apps.stable_diffusion.src.utils.sd_annotation import sd_model_annotation
import sys, functools, operator
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    load_pipeline_from_original_stable_diffusion_ckpt,
)


def get_vmfb_path_name(model_name):
    device = (
        args.device
        if "://" not in args.device
        else "-".join(args.device.split("://"))
    )
    extended_name = "{}_{}".format(model_name, device)
    vmfb_path = os.path.join(os.getcwd(), extended_name + ".vmfb")
    return [vmfb_path, extended_name]


def _compile_module(shark_module, model_name, extra_args=[]):
    if args.load_vmfb or args.save_vmfb:
        [vmfb_path, extended_name] = get_vmfb_path_name(model_name)
        if args.load_vmfb and os.path.isfile(vmfb_path) and not args.save_vmfb:
            print(f"loading existing vmfb from: {vmfb_path}")
            shark_module.load_module(vmfb_path, extra_args=extra_args)
        else:
            if args.save_vmfb:
                print("Saving to {}".format(vmfb_path))
            else:
                print(
                    "No vmfb found. Compiling and saving to {}".format(
                        vmfb_path
                    )
                )
            path = shark_module.save_module(
                os.getcwd(), extended_name, extra_args
            )
            shark_module.load_module(path, extra_args=extra_args)
    else:
        shark_module.compile(extra_args)
    return shark_module


# Downloads the model from shark_tank and returns the shark_module.
def get_shark_model(tank_url, model_name, extra_args=[]):
    from shark.shark_downloader import download_model
    from shark.parser import shark_args

    # Set local shark_tank cache directory.
    shark_args.local_tank_cache = args.local_tank_cache
    if "cuda" in args.device:
        shark_args.enable_tf32 = True

    mlir_model, func_name, inputs, golden_out = download_model(
        model_name,
        tank_url=tank_url,
        frontend="torch",
    )
    shark_module = SharkInference(
        mlir_model, device=args.device, mlir_dialect="linalg"
    )
    return _compile_module(shark_module, model_name, extra_args)


# Converts the torch-module into a shark_module.
def compile_through_fx(
    model,
    inputs,
    model_name,
    is_f16=False,
    f16_input_mask=None,
    use_tuned=False,
    extra_args=[],
):
    from shark.parser import shark_args

    if "cuda" in args.device:
        shark_args.enable_tf32 = True

    mlir_module, func_name = import_with_fx(
        model, inputs, is_f16, f16_input_mask
    )

    if use_tuned:
        tuned_model_path = f"{args.annotation_output}/{model_name}_torch.mlir"
        if not os.path.exists(tuned_model_path):
            if "vae" in model_name.split("_")[0]:
                args.annotation_model = "vae"

            tuned_model, tuned_model_path = sd_model_annotation(
                mlir_module, model_name
            )
            del mlir_module, tuned_model
            gc.collect()

        with open(tuned_model_path, "rb") as f:
            mlir_module = f.read()
            f.close()

    shark_module = SharkInference(
        mlir_module,
        device=args.device,
        mlir_dialect="linalg",
    )
    return _compile_module(shark_module, model_name, extra_args)


def set_iree_runtime_flags():
    vulkan_runtime_flags = [
        f"--vulkan_large_heap_block_size={args.vulkan_large_heap_block_size}",
        f"--vulkan_validation_layers={'true' if args.vulkan_validation_layers else 'false'}",
    ]
    if args.enable_rgp:
        vulkan_runtime_flags += [
            f"--enable_rgp=true",
            f"--vulkan_debug_utils=true",
        ]
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
        key_combination (int, optional): choice for mapping value for device name.
        1 : path
        2 : name
        3 : (name, path)
        Defaults to 3.
    Returns:
        dict: map to possible device names user can input mapped to desired combination of name/path.
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
            return (dev_dict["name"], f"{driver}://{dev_dict['path']}")

    # mapping driver name to default device (driver://0)
    device_map[f"{driver}"] = get_output_value(device_list[0])
    for i, device in enumerate(device_list):
        # mapping with index
        device_map[f"{driver}://{i}"] = get_output_value(device)
        # mapping with full path
        device_map[f"{driver}://{device['path']}"] = get_output_value(device)
    return device_map


def map_device_to_name_path(device, key_combination=3):
    """Gives the appropriate device data (supported name/path) for user selected execution device
    Args:
        device (str): user
        key_combination (int, optional): choice for mapping value for device name.
        1 : path
        2 : name
        3 : (name, path)
        Defaults to 3.
    Raises:
        ValueError:
    Returns:
        str / tuple: returns the mapping str or tuple of mapping str for the device depending on key_combination value
    """
    driver = device.split("://")[0]
    device_map = get_device_mapping(driver, key_combination)
    try:
        device_mapping = device_map[device]
    except KeyError:
        raise ValueError(f"Device '{device}' is not a valid device.")
    return device_mapping


def set_init_device_flags():
    if "vulkan" in args.device:
        # set runtime flags for vulkan.
        set_iree_runtime_flags()

        # set triple flag to avoid multiple calls to get_vulkan_triple_flag
        device_name, args.device = map_device_to_name_path(args.device)
        if not args.iree_vulkan_target_triple:
            triple = get_vulkan_target_triple(device_name)
            if triple is not None:
                args.iree_vulkan_target_triple = triple
        print(
            f"Found device {device_name}. Using target triple {args.iree_vulkan_target_triple}."
        )
    elif "cuda" in args.device:
        args.device = "cuda"
    elif "cpu" in args.device:
        args.device = "cpu"

    # set max_length based on availability.
    if args.hf_model_id in [
        "Linaqruf/anything-v3.0",
        "wavymulder/Analog-Diffusion",
        "dreamlike-art/dreamlike-diffusion-1.0",
    ]:
        args.max_length = 77
    elif args.hf_model_id == "prompthero/openjourney":
        args.max_length = 64

    # Use tuned models in the case of fp16, vulkan rdna3 or cuda sm devices.
    if (
        args.hf_model_id == "prompthero/openjourney"
        or args.ckpt_loc != ""
        or args.precision != "fp16"
        or args.height != 512
        or args.width != 512
        or args.batch_size != 1
        or ("vulkan" not in args.device and "cuda" not in args.device)
    ):
        args.use_tuned = False

    elif (
        "vulkan" in args.device
        and "rdna3" not in args.iree_vulkan_target_triple
    ):
        args.use_tuned = False

    elif "cuda" in args.device and get_cuda_sm_cc() not in [
        "sm_80",
        "sm_84",
        "sm_86",
    ]:
        args.use_tuned = False

    elif args.use_base_vae and args.hf_model_id not in [
        "stabilityai/stable-diffusion-2-1-base",
        "CompVis/stable-diffusion-v1-4",
    ]:
        args.use_tuned = False

    if args.use_tuned:
        print(f"Using tuned models for {args.hf_model_id}/fp16/{args.device}.")
    else:
        print("Tuned models are currently not supported for this setting.")

    # set import_mlir to True for unuploaded models.
    if args.ckpt_loc != "":
        args.import_mlir = True

    elif args.hf_model_id not in [
        "Linaqruf/anything-v3.0",
        "dreamlike-art/dreamlike-diffusion-1.0",
        "prompthero/openjourney",
        "wavymulder/Analog-Diffusion",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-2-1-base",
        "CompVis/stable-diffusion-v1-4",
    ]:
        args.import_mlir = True

    elif args.height != 512 or args.width != 512 or args.batch_size != 1:
        args.import_mlir = True


# Utility to get list of devices available.
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
            for i, device in enumerate(device_list_dict):
                device_list.append(f"{device['name']} => {driver_name}://{i}")
        return device_list

    set_iree_runtime_flags()

    available_devices = []
    vulkan_devices = get_devices_by_name("vulkan")
    available_devices.extend(vulkan_devices)
    cuda_devices = get_devices_by_name("cuda")
    available_devices.extend(cuda_devices)
    available_devices.append("cpu")
    return available_devices


def disk_space_check(path, lim=20):
    from shutil import disk_usage

    du = disk_usage(path)
    free = du.free / (1024 * 1024 * 1024)
    if free <= lim:
        print(f"[WARNING] Only {free:.2f}GB space available in {path}.")


def get_opt_flags(model, precision="fp16"):
    iree_flags = []
    is_tuned = "tuned" if args.use_tuned else "untuned"
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )

    # Disable bindings fusion to work with moltenVK.
    if sys.platform == "darwin":
        iree_flags.append("-iree-stream-fuse-binding=false")

    if "default_compilation_flags" in opt_flags[model][is_tuned][precision]:
        iree_flags += opt_flags[model][is_tuned][precision][
            "default_compilation_flags"
        ]

    if "specified_compilation_flags" in opt_flags[model][is_tuned][precision]:
        device = (
            args.device
            if "://" not in args.device
            else args.device.split("://")[0]
        )
        if (
            device
            not in opt_flags[model][is_tuned][precision][
                "specified_compilation_flags"
            ]
        ):
            device = "default_device"
        iree_flags += opt_flags[model][is_tuned][precision][
            "specified_compilation_flags"
        ][device]
    return iree_flags


def get_path_to_diffusers_checkpoint(custom_weights):
    path = Path(custom_weights)
    diffusers_path = path.parent.absolute()
    diffusers_directory_name = path.stem
    complete_path_to_diffusers = diffusers_path / diffusers_directory_name
    complete_path_to_diffusers.mkdir(parents=True, exist_ok=True)
    path_to_diffusers = complete_path_to_diffusers.as_posix()
    return path_to_diffusers


def preprocessCKPT(custom_weights):
    path_to_diffusers = get_path_to_diffusers_checkpoint(custom_weights)
    if next(Path(path_to_diffusers).iterdir(), None):
        print("Checkpoint already loaded at : ", path_to_diffusers)
        return
    else:
        print(
            "Diffusers' checkpoint will be identified here : ",
            path_to_diffusers,
        )
    from_safetensors = (
        True if custom_weights.lower().endswith(".safetensors") else False
    )
    # EMA weights usually yield higher quality images for inference but non-EMA weights have
    # been yielding better results in our case.
    # TODO: Add an option `--ema` (`--no-ema`) for users to specify if they want to go for EMA
    #       weight extraction or not.
    extract_ema = False
    print(
        "Loading diffusers' pipeline from original stable diffusion checkpoint"
    )
    pipe = load_pipeline_from_original_stable_diffusion_ckpt(
        checkpoint_path=custom_weights,
        extract_ema=extract_ema,
        from_safetensors=from_safetensors,
    )
    pipe.save_pretrained(path_to_diffusers)
    print("Loading complete")


def load_vmfb(vmfb_path, model, precision):
    model = "vae" if "base_vae" in model else model
    precision = "fp32" if "clip" in model else precision
    extra_args = get_opt_flags(model, precision)
    shark_module = SharkInference(mlir_module=None, device=args.device)
    shark_module.load_module(vmfb_path, extra_args=extra_args)
    return shark_module


# This utility returns vmfbs of Clip, Unet and Vae, in case all three of them
# are present; deletes them otherwise.
def fetch_or_delete_vmfbs(basic_model_name, use_base_vae, precision="fp32"):
    model_name = ["clip", "unet", "base_vae" if use_base_vae else "vae"]
    vmfb_path = [
        get_vmfb_path_name(model + basic_model_name)[0] for model in model_name
    ]
    vmfb_present = [os.path.isfile(vmfb) for vmfb in vmfb_path]
    all_vmfb_present = functools.reduce(operator.__and__, vmfb_present)
    compiled_models = [None] * 3
    # We need to delete vmfbs only if some of the models were compiled.
    if not all_vmfb_present:
        for i in range(len(vmfb_path)):
            if vmfb_present[i]:
                os.remove(vmfb_path[i])
                print("Deleted: ", vmfb_path[i])
    else:
        for i in range(len(vmfb_path)):
            compiled_models[i] = load_vmfb(
                vmfb_path[i], model_name[i], precision
            )
    return compiled_models


# Generate and return a new seed if the provided one is not in the supported range (including -1)
def sanitize_seed(seed):
    uint32_info = np.iinfo(np.uint32)
    uint32_min, uint32_max = uint32_info.min, uint32_info.max
    if seed < uint32_min or seed >= uint32_max:
        seed = randint(uint32_min, uint32_max)
    return seed
