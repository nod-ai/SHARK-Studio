import os
import gc
import json
import re
from PIL import PngImagePlugin
from PIL import Image
from datetime import datetime as dt
from csv import DictWriter
from pathlib import Path
import numpy as np
from random import (
    randint,
    seed as seed_random,
    getstate as random_getstate,
    setstate as random_setstate,
)
import tempfile
import torch
from safetensors.torch import load_file
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from shark.iree_utils.vulkan_utils import (
    set_iree_vulkan_runtime_flags,
    get_vulkan_target_triple,
    get_iree_vulkan_runtime_flags,
)
from shark.iree_utils.metal_utils import get_metal_target_triple
from shark.iree_utils.gpu_utils import get_cuda_sm_cc, get_iree_rocm_args
from apps.stable_diffusion.src.utils.stable_args import args
from apps.stable_diffusion.src.utils.resources import opt_flags
from apps.stable_diffusion.src.utils.sd_annotation import sd_model_annotation
import sys
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
    create_vae_diffusers_config,
    convert_ldm_vae_checkpoint,
)
import requests
from io import BytesIO
from omegaconf import OmegaConf
from cpuinfo import get_cpu_info


def get_extended_name(model_name):
    device = args.device.split("://", 1)[0]
    extended_name = "{}_{}".format(model_name, device)
    return extended_name


def get_vmfb_path_name(model_name):
    vmfb_path = os.path.join(os.getcwd(), model_name + ".vmfb")
    return vmfb_path


def _load_vmfb(shark_module, vmfb_path, model, precision):
    model = "vae" if "base_vae" in model or "vae_encode" in model else model
    model = "unet" if "stencil" in model else model
    model = "unet" if "unet512" in model else model
    precision = "fp32" if "clip" in model else precision
    extra_args = get_opt_flags(model, precision)
    shark_module.load_module(vmfb_path, extra_args=extra_args)
    return shark_module


def _compile_module(shark_module, model_name, extra_args=[]):
    if args.load_vmfb or args.save_vmfb:
        vmfb_path = get_vmfb_path_name(model_name)
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
                os.getcwd(), model_name, extra_args, debug=args.compile_debug
            )
            shark_module.load_module(path, extra_args=extra_args)
    else:
        shark_module.compile(extra_args)
    return shark_module


# Downloads the model from shark_tank and returns the shark_module.
def get_shark_model(tank_url, model_name, extra_args=None):
    if extra_args is None:
        extra_args = []
    from shark.parser import shark_args

    # Set local shark_tank cache directory.
    shark_args.local_tank_cache = args.local_tank_cache
    from shark.shark_downloader import download_model

    if "cuda" in args.device:
        shark_args.enable_tf32 = True

    mlir_model, func_name, inputs, golden_out = download_model(
        model_name,
        tank_url=tank_url,
        frontend="torch",
    )
    shark_module = SharkInference(
        mlir_model, device=args.device, mlir_dialect="tm_tensor"
    )
    return _compile_module(shark_module, model_name, extra_args)


# Converts the torch-module into a shark_module.
def compile_through_fx(
    model,
    inputs,
    extended_model_name,
    is_f16=False,
    f16_input_mask=None,
    use_tuned=False,
    save_dir=tempfile.gettempdir(),
    debug=False,
    generate_vmfb=True,
    extra_args=None,
    base_model_id=None,
    model_name=None,
    precision=None,
    return_mlir=False,
    device=None,
):
    if extra_args is None:
        extra_args = []
    if not return_mlir and model_name is not None:
        vmfb_path = get_vmfb_path_name(extended_model_name)
        if os.path.isfile(vmfb_path):
            shark_module = SharkInference(mlir_module=None, device=args.device)
            return (
                _load_vmfb(shark_module, vmfb_path, model_name, precision),
                None,
            )

    from shark.parser import shark_args

    if "cuda" in args.device:
        shark_args.enable_tf32 = True

    (
        mlir_module,
        func_name,
    ) = import_with_fx(
        model=model,
        inputs=inputs,
        is_f16=is_f16,
        f16_input_mask=f16_input_mask,
        debug=debug,
        model_name=extended_model_name,
        save_dir=save_dir,
    )
    if use_tuned:
        if "vae" in extended_model_name.split("_")[0]:
            args.annotation_model = "vae"
        if (
            "unet" in model_name.split("_")[0]
            or "unet_512" in model_name.split("_")[0]
        ):
            args.annotation_model = "unet"
        mlir_module = sd_model_annotation(
            mlir_module, extended_model_name, base_model_id
        )

    shark_module = SharkInference(
        mlir_module,
        device=args.device if device is None else device,
        mlir_dialect="tm_tensor",
    )
    if generate_vmfb:
        return (
            _compile_module(shark_module, extended_model_name, extra_args),
            mlir_module,
        )

    del mlir_module
    gc.collect()


def set_iree_runtime_flags():
    vulkan_runtime_flags = get_iree_vulkan_runtime_flags()
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
            f"Found device {device_name}. Using target triple "
            f"{args.iree_vulkan_target_triple}."
        )
    elif "cuda" in args.device:
        args.device = "cuda"
    elif "metal" in args.device:
        device_name, args.device = map_device_to_name_path(args.device)
        if not args.iree_metal_target_platform:
            triple = get_metal_target_triple(device_name)
            if triple is not None:
                args.iree_metal_target_platform = triple.split("-")[-1]
        print(
            f"Found device {device_name}. Using target triple "
            f"{args.iree_metal_target_platform}."
        )
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
    if args.ckpt_loc != "":
        base_model_id = fetch_and_update_base_model_id(args.ckpt_loc)
    else:
        base_model_id = fetch_and_update_base_model_id(args.hf_model_id)
        if base_model_id == "":
            base_model_id = args.hf_model_id

    if (
        args.precision != "fp16"
        or args.height not in [512, 768]
        or (args.height == 512 and args.width not in [512, 768])
        or (args.height == 768 and args.width not in [512, 768])
        or args.batch_size != 1
        or ("vulkan" not in args.device and "cuda" not in args.device)
    ):
        args.use_tuned = False

    elif (
        args.height != args.width
        and "rdna2" in args.iree_vulkan_target_triple
        and base_model_id
        not in [
            "CompVis/stable-diffusion-v1-4",
            "runwayml/stable-diffusion-v1-5",
        ]
    ):
        args.use_tuned = False

    elif base_model_id not in [
        "Linaqruf/anything-v3.0",
        "dreamlike-art/dreamlike-diffusion-1.0",
        "prompthero/openjourney",
        "wavymulder/Analog-Diffusion",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-2-1-base",
        "CompVis/stable-diffusion-v1-4",
        "runwayml/stable-diffusion-v1-5",
        "runwayml/stable-diffusion-inpainting",
        "stabilityai/stable-diffusion-2-inpainting",
    ]:
        args.use_tuned = False

    elif "vulkan" in args.device and not any(
        x in args.iree_vulkan_target_triple for x in ["rdna2", "rdna3"]
    ):
        args.use_tuned = False

    elif "cuda" in args.device and get_cuda_sm_cc() not in ["sm_80", "sm_89"]:
        args.use_tuned = False

    elif args.use_base_vae and args.hf_model_id not in [
        "stabilityai/stable-diffusion-2-1-base",
        "CompVis/stable-diffusion-v1-4",
    ]:
        args.use_tuned = False

    elif (
        args.height == 768
        and args.width == 768
        and (
            base_model_id
            not in [
                "stabilityai/stable-diffusion-2-1",
                "stabilityai/stable-diffusion-2-1-base",
            ]
            or "rdna" not in args.iree_vulkan_target_triple
        )
    ):
        args.use_tuned = False

    elif "rdna2" in args.iree_vulkan_target_triple and (
        base_model_id
        not in [
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-2-1-base",
            "CompVis/stable-diffusion-v1-4",
        ]
    ):
        args.use_tuned = False

    if args.use_tuned:
        print(
            f"Using tuned models for {base_model_id}(fp16) on "
            f"device {args.device}."
        )
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

    elif args.use_tuned and args.hf_model_id in [
        "dreamlike-art/dreamlike-diffusion-1.0",
        "prompthero/openjourney",
        "stabilityai/stable-diffusion-2-1",
    ]:
        args.import_mlir = True

    elif (
        args.use_tuned
        and "vulkan" in args.device
        and "rdna2" in args.iree_vulkan_target_triple
    ):
        args.import_mlir = True

    elif (
        args.use_tuned
        and "cuda" in args.device
        and get_cuda_sm_cc() == "sm_89"
    ):
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
                    device_list.append(f"{device_name} => {driver_name}://{i}")
        return device_list

    set_iree_runtime_flags()

    available_devices = []
    from shark.iree_utils._common import run_cmd
    from shark.iree_utils.vulkan_utils import (
        get_all_vulkan_devices,
    )

    vulkaninfo_list = get_all_vulkan_devices()
    vulkan_devices = []
    id = 0
    for device in vulkaninfo_list:
        vulkan_devices.append(
            f"{device.split('=')[1].strip()} => vulkan://{id}"
        )
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
    if "rocm" in args.device:
        rocm_args = get_iree_rocm_args()
        iree_flags.extend(rocm_args)
        print(iree_flags)
    if args.iree_constant_folding == False:
        iree_flags.append("--iree-opt-const-expr-hoisting=False")
        iree_flags.append(
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807"
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


def get_path_stem(path):
    path = Path(path)
    return path.stem


def get_path_to_diffusers_checkpoint(custom_weights):
    path = Path(custom_weights)
    diffusers_path = path.parent.absolute()
    diffusers_directory_name = os.path.join("diffusers", path.stem)
    complete_path_to_diffusers = diffusers_path / diffusers_directory_name
    complete_path_to_diffusers.mkdir(parents=True, exist_ok=True)
    path_to_diffusers = complete_path_to_diffusers.as_posix()
    return path_to_diffusers


def preprocessCKPT(custom_weights, is_inpaint=False):
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
    # EMA weights usually yield higher quality images for inference but
    # non-EMA weights have been yielding better results in our case.
    # TODO: Add an option `--ema` (`--no-ema`) for users to specify if
    #  they want to go for EMA weight extraction or not.
    extract_ema = False
    print(
        "Loading diffusers' pipeline from original stable diffusion checkpoint"
    )
    num_in_channels = 9 if is_inpaint else 4
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=custom_weights,
        extract_ema=extract_ema,
        from_safetensors=from_safetensors,
        num_in_channels=num_in_channels,
    )
    pipe.save_pretrained(path_to_diffusers)
    print("Loading complete")


def convert_original_vae(vae_checkpoint):
    vae_state_dict = {}
    for key in list(vae_checkpoint.keys()):
        vae_state_dict["first_stage_model." + key] = vae_checkpoint.get(key)

    config_url = (
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/"
        "main/configs/stable-diffusion/v1-inference.yaml"
    )
    original_config_file = BytesIO(requests.get(config_url).content)
    original_config = OmegaConf.load(original_config_file)
    vae_config = create_vae_diffusers_config(original_config, image_size=512)

    converted_vae_checkpoint = convert_ldm_vae_checkpoint(
        vae_state_dict, vae_config
    )
    return converted_vae_checkpoint


def processLoRA(model, use_lora, splitting_prefix):
    state_dict = ""
    if ".safetensors" in use_lora:
        state_dict = load_file(use_lora)
    else:
        state_dict = torch.load(use_lora)
    alpha = 0.75
    visited = []

    # directly update weight in model
    process_unet = "te" not in splitting_prefix
    for key in state_dict:
        if ".alpha" in key or key in visited:
            continue

        curr_layer = model
        if ("text" not in key and process_unet) or (
            "text" in key and not process_unet
        ):
            layer_infos = (
                key.split(".")[0].split(splitting_prefix)[-1].split("_")
            )
        else:
            continue

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = (
                state_dict[pair_keys[0]]
                .squeeze(3)
                .squeeze(2)
                .to(torch.float32)
            )
            weight_down = (
                state_dict[pair_keys[1]]
                .squeeze(3)
                .squeeze(2)
                .to(torch.float32)
            )
            curr_layer.weight.data += alpha * torch.mm(
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
        # update visited list
        for item in pair_keys:
            visited.append(item)
    return model


def update_lora_weight_for_unet(unet, use_lora):
    extensions = [".bin", ".safetensors", ".pt"]
    if not any([extension in use_lora for extension in extensions]):
        # We assume if it is a HF ID with standalone LoRA weights.
        unet.load_attn_procs(use_lora)
        return unet

    main_file_name = get_path_stem(use_lora)
    if ".bin" in use_lora:
        main_file_name += ".bin"
    elif ".safetensors" in use_lora:
        main_file_name += ".safetensors"
    elif ".pt" in use_lora:
        main_file_name += ".pt"
    else:
        sys.exit("Only .bin and .safetensors format for LoRA is supported")

    try:
        dir_name = os.path.dirname(use_lora)
        unet.load_attn_procs(dir_name, weight_name=main_file_name)
        return unet
    except:
        return processLoRA(unet, use_lora, "lora_unet_")


def update_lora_weight(model, use_lora, model_name):
    if "unet" in model_name:
        return update_lora_weight_for_unet(model, use_lora)
    try:
        return processLoRA(model, use_lora, "lora_te_")
    except:
        return None


# `fetch_and_update_base_model_id` is a resource utility function which
# helps to maintain mapping of the model to run with its base model.
# If `base_model` is "", then this function tries to fetch the base model
# info for the `model_to_run`.
def fetch_and_update_base_model_id(model_to_run, base_model=""):
    variants_path = os.path.join(os.getcwd(), "variants.json")
    data = {model_to_run: base_model}
    json_data = {}
    if os.path.exists(variants_path):
        with open(variants_path, "r", encoding="utf-8") as jsonFile:
            json_data = json.load(jsonFile)
            # Return with base_model's info if base_model is "".
            if base_model == "":
                if model_to_run in json_data:
                    base_model = json_data[model_to_run]
                return base_model
    elif base_model == "":
        return base_model
    # Update JSON data to contain an entry mapping model_to_run with
    # base_model.
    json_data.update(data)
    with open(variants_path, "w", encoding="utf-8") as jsonFile:
        json.dump(json_data, jsonFile)


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

    if isinstance(seed_input, list) and all(
        type(seed) is int for seed in seed_input
    ):
        return seed_input

    raise TypeError(
        "Seed input must be an integer or an array of integers in JSON format"
    )


# Generate a set of seeds from an input expression for batch_count batches,
# optionally using that input as the rng seed for any randomly generated seeds.
def batch_seeds(
    seed_input: str | list | int, batch_count: int, repeatable=False
):
    # turn the input into a list if possible
    seeds = parse_seed_input(seed_input)

    # slice or pad the list to be of batch_count length
    seeds = seeds[:batch_count] + [-1] * (batch_count - len(seeds))

    if repeatable:
        # set seed for the rng based on what we have so far
        saved_random_state = random_getstate()
        if all(seed < 0 for seed in seeds):
            seeds[0] = sanitize_seed(seeds[0])
        seed_random(str(seeds))

    # generate any seeds that are unspecified
    seeds = [sanitize_seed(seed) for seed in seeds]

    if repeatable:
        # reset the rng back to normal
        random_setstate(saved_random_state)

    return seeds


# clear all the cached objects to recompile cleanly.
def clear_all():
    print("CLEARING ALL, EXPECT SEVERAL MINUTES TO RECOMPILE")
    from glob import glob
    import shutil

    vmfbs = glob(os.path.join(os.getcwd(), "*.vmfb"))
    for vmfb in vmfbs:
        if os.path.exists(vmfb):
            os.remove(vmfb)
    # Temporary workaround of deleting yaml files to incorporate
    # diffusers' pipeline.
    # TODO: Remove this once we have better weight updation logic.
    inference_yaml = ["v2-inference-v.yaml", "v1-inference.yaml"]
    for yaml in inference_yaml:
        if os.path.exists(yaml):
            os.remove(yaml)
    home = os.path.expanduser("~")
    if os.name == "nt":  # Windows
        appdata = os.getenv("LOCALAPPDATA")
        shutil.rmtree(os.path.join(appdata, "AMD/VkCache"), ignore_errors=True)
        shutil.rmtree(
            os.path.join(home, ".local/shark_tank"), ignore_errors=True
        )
    elif os.name == "unix":
        shutil.rmtree(os.path.join(home, ".cache/AMD/VkCache"))
        shutil.rmtree(os.path.join(home, ".local/shark_tank"))
    if args.local_tank_cache != "":
        shutil.rmtree(args.local_tank_cache)


def get_generated_imgs_path() -> Path:
    return Path(
        args.output_dir if args.output_dir else Path.cwd(), "generated_imgs"
    )


def get_generated_imgs_todays_subdir() -> str:
    return dt.now().strftime("%Y%m%d")


# save output images and the inputs corresponding to it.
def save_output_img(output_img, img_seed, extra_info=None):
    if extra_info is None:
        extra_info = {}
    generated_imgs_path = Path(
        get_generated_imgs_path(), get_generated_imgs_todays_subdir()
    )
    generated_imgs_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(generated_imgs_path, "imgs_details.csv")

    prompt_slice = re.sub("[^a-zA-Z0-9]", "_", args.prompts[0][:15])
    out_img_name = f"{dt.now().strftime('%H%M%S')}_{prompt_slice}_{img_seed}"

    img_model = args.hf_model_id
    if args.ckpt_loc:
        img_model = Path(os.path.basename(args.ckpt_loc)).stem

    img_vae = None
    if args.custom_vae:
        img_vae = Path(os.path.basename(args.custom_vae)).stem

    img_lora = None
    if args.use_lora:
        img_lora = Path(os.path.basename(args.use_lora)).stem

    if args.output_img_format == "jpg":
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.jpg")
        output_img.save(out_img_path, quality=95, subsampling=0)
    else:
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.png")
        pngInfo = PngImagePlugin.PngInfo()

        if args.write_metadata_to_png:
            pngInfo.add_text(
                "parameters",
                f"{args.prompts[0]}"
                f"\nNegative prompt: {args.negative_prompts[0]}"
                f"\nSteps: {args.steps},"
                f"Sampler: {args.scheduler}, "
                f"CFG scale: {args.guidance_scale}, "
                f"Seed: {img_seed},"
                f"Size: {args.width}x{args.height}, "
                f"Model: {img_model}, "
                f"VAE: {img_vae}, "
                f"LoRA: {img_lora}",
            )

        output_img.save(out_img_path, "PNG", pnginfo=pngInfo)

        if args.output_img_format not in ["png", "jpg"]:
            print(
                f"[ERROR] Format {args.output_img_format} is not "
                f"supported yet. Image saved as png instead."
                f"Supported formats: png / jpg"
            )

    # To be as low-impact as possible to the existing CSV format, we append
    # "VAE" and "LORA" to the end. However, it does not fit the hierarchy of
    # importance for each data point. Something to consider.
    new_entry = {
        "VARIANT": img_model,
        "SCHEDULER": args.scheduler,
        "PROMPT": args.prompts[0],
        "NEG_PROMPT": args.negative_prompts[0],
        "SEED": img_seed,
        "CFG_SCALE": args.guidance_scale,
        "PRECISION": args.precision,
        "STEPS": args.steps,
        "HEIGHT": args.height,
        "WIDTH": args.width,
        "MAX_LENGTH": args.max_length,
        "OUTPUT": out_img_path,
        "VAE": img_vae,
        "LORA": img_lora,
    }

    new_entry.update(extra_info)

    csv_mode = "a" if os.path.isfile(csv_path) else "w"
    with open(csv_path, csv_mode, encoding="utf-8") as csv_obj:
        dictwriter_obj = DictWriter(csv_obj, fieldnames=list(new_entry.keys()))
        if csv_mode == "w":
            dictwriter_obj.writeheader()
        dictwriter_obj.writerow(new_entry)
        csv_obj.close()

    if args.save_metadata_to_json:
        del new_entry["OUTPUT"]
        json_path = Path(generated_imgs_path, f"{out_img_name}.json")
        with open(json_path, "w") as f:
            json.dump(new_entry, f, indent=4)


def get_generation_text_info(seeds, device):
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += (
        f"\nmodel_id={args.hf_model_id}, " f"ckpt_loc={args.ckpt_loc}"
    )
    text_output += f"\nscheduler={args.scheduler}, " f"device={device}"
    text_output += (
        f"\nsteps={args.steps}, "
        f"guidance_scale={args.guidance_scale}, "
        f"seed={seeds}"
    )
    text_output += (
        f"\nsize={args.height}x{args.width}, "
        f"batch_count={args.batch_count}, "
        f"batch_size={args.batch_size}, "
        f"max_length={args.max_length}"
    )

    return text_output


# For stencil, the input image can be of any size, but we need to ensure that
# it conforms with our model constraints :-
#   Both width and height should be in the range of [128, 768] and multiple of 8.
# This utility function performs the transformation on the input image while
# also maintaining the aspect ratio before sending it to the stencil pipeline.
def resize_stencil(image: Image.Image):
    width, height = image.size
    aspect_ratio = width / height
    min_size = min(width, height)
    if min_size < 128:
        n_size = 128
        if width == min_size:
            width = n_size
            height = n_size / aspect_ratio
        else:
            height = n_size
            width = n_size * aspect_ratio
    width = int(width)
    height = int(height)
    n_width = width // 8
    n_height = height // 8
    n_width *= 8
    n_height *= 8

    min_size = min(width, height)
    if min_size > 768:
        n_size = 768
        if width == min_size:
            height = n_size
            width = n_size * aspect_ratio
        else:
            width = n_size
            height = n_size / aspect_ratio
    width = int(width)
    height = int(height)
    n_width = width // 8
    n_height = height // 8
    n_width *= 8
    n_height *= 8
    new_image = image.resize((n_width, n_height))
    return new_image, n_width, n_height
