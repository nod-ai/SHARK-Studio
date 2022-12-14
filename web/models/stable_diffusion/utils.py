import os

import torch
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from shark.iree_utils.vulkan_utils import set_iree_vulkan_runtime_flags


def set_iree_runtime_flags(args):
    vulkan_runtime_flags = [
        f"--vulkan_large_heap_block_size={args.vulkan_large_heap_block_size}",
    ]
    if "vulkan" in args.device:
        set_iree_vulkan_runtime_flags(flags=vulkan_runtime_flags)

    return


def _compile_module(args, shark_module, model_name, extra_args=[]):
    device = (
        args.device
        if "://" not in args.device
        else "-".join(args.device.split("://"))
    )
    extended_name = "{}_{}".format(model_name, device)
    if args.cache:
        vmfb_path = os.path.join(os.getcwd(), extended_name + ".vmfb")
        if os.path.isfile(vmfb_path):
            print("Loading flatbuffer from {}".format(vmfb_path))
            shark_module.load_module(vmfb_path)
            return shark_module
        print("No vmfb found. Compiling and saving to {}".format(vmfb_path))
    path = shark_module.save_module(os.getcwd(), extended_name, extra_args)
    shark_module.load_module(path)
    return shark_module


# Downloads the model from shark_tank and returns the shark_module.
def get_shark_model(args, tank_url, model_name, extra_args=[]):
    from shark.shark_downloader import download_model
    from shark.parser import shark_args

    # Set local shark_tank cache directory.
    shark_args.local_tank_cache = args.local_tank_cache

    mlir_model, func_name, inputs, golden_out = download_model(
        model_name, tank_url=tank_url, frontend="torch"
    )
    shark_module = SharkInference(
        mlir_model, func_name, device=args.device, mlir_dialect="linalg"
    )
    return _compile_module(args, shark_module, model_name, extra_args)


# Converts the torch-module into shark_module.
def compile_through_fx(args, model, inputs, model_name, extra_args=[]):

    mlir_module, func_name = import_with_fx(model, inputs)

    shark_module = SharkInference(
        mlir_module,
        func_name,
        device=args.device,
        mlir_dialect="linalg",
    )

    return _compile_module(args, shark_module, model_name, extra_args)
