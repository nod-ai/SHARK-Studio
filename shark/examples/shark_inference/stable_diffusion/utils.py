import os

import torch
from shark.shark_inference import SharkInference
from stable_args import args
from shark.shark_importer import import_with_fx
from shark.iree_utils.vulkan_utils import set_iree_vulkan_runtime_flags
from shark.iree_utils._common import map_device_to_path


def _compile_module(shark_module, model_name, extra_args=[]):
    if args.load_vmfb or args.save_vmfb:
        device = (
            args.device
            if "://" not in args.device
            else "-".join(args.device.split("://"))
        )
        extended_name = "{}_{}".format(model_name, device)
        vmfb_path = os.path.join(os.getcwd(), extended_name + ".vmfb")
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

    mlir_model, func_name, inputs, golden_out = download_model(
        model_name,
        tank_url=tank_url,
        frontend="torch",
    )
    shark_module = SharkInference(
        mlir_model, func_name, device=args.device, mlir_dialect="linalg"
    )
    return _compile_module(shark_module, model_name, extra_args)


# Converts the torch-module into a shark_module.
def compile_through_fx(model, inputs, model_name, extra_args=[]):

    mlir_module, func_name = import_with_fx(model, inputs)

    shark_module = SharkInference(
        mlir_module,
        func_name,
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
    if "vulkan" in args.device:
        set_iree_vulkan_runtime_flags(flags=vulkan_runtime_flags)

    return


def make_qualified_device_name():
    # modify device name to be fully qualified device name
    # of the format driver://path
    # supported for vulkan as of now

    if "vulkan" in args.device:
        args.device = map_device_to_path(args.device)
