import os
from pathlib import Path
from shark_tuner.codegen_tuner import SharkCodegenTuner
from shark_tuner.iree_utils import (
    dump_dispatches,
    create_context,
    export_module_to_mlir_file,
)
from shark_tuner.model_annotation import model_annotation
from apps.stable_diffusion.src.utils.stable_args import args
from apps.stable_diffusion.src.utils.utils import set_init_device_flags
from apps.stable_diffusion.src.utils.sd_annotation import (
    get_device_args,
    load_winograd_configs,
)
from apps.stable_diffusion.src.models import SharkifyStableDiffusionModel


def load_mlir_module():
    if "upscaler" in args.hf_model_id:
        is_upscaler = True
    else:
        is_upscaler = False
    sd_model = SharkifyStableDiffusionModel(
        args.hf_model_id,
        args.ckpt_loc,
        args.custom_vae,
        args.precision,
        max_len=args.max_length,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        use_base_vae=args.use_base_vae,
        is_upscaler=is_upscaler,
        use_tuned=False,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        return_mlir=True,
    )

    if args.annotation_model == "unet":
        mlir_module = sd_model.unet()
        model_name = sd_model.model_name["unet"]
    elif args.annotation_model == "vae":
        mlir_module = sd_model.vae()
        model_name = sd_model.model_name["vae"]
    else:
        raise ValueError(
            f"{args.annotation_model} is not supported for tuning."
        )

    return mlir_module, model_name


def main():
    args.use_tuned = False
    set_init_device_flags()
    mlir_module, model_name = load_mlir_module()

    # Get device and device specific arguments
    device, device_spec_args = get_device_args()
    device_spec = ""
    vulkan_target_triple = ""
    if device_spec_args:
        device_spec = device_spec_args[-1].split("=")[-1].strip()
        if device == "vulkan":
            vulkan_target_triple = device_spec
            device_spec = device_spec.split("-")[0]

    # Add winograd annotation for vulkan device
    use_winograd = (
        True
        if device == "vulkan" and args.annotation_model in ["unet", "vae"]
        else False
    )
    winograd_config = (
        load_winograd_configs()
        if device == "vulkan" and args.annotation_model in ["unet", "vae"]
        else ""
    )
    with create_context() as ctx:
        input_module = model_annotation(
            ctx,
            input_contents=mlir_module,
            config_path=winograd_config,
            search_op="conv",
            winograd=use_winograd,
        )

    # Dump model dispatches
    generates_dir = Path.home() / "tmp"
    if not os.path.exists(generates_dir):
        os.makedirs(generates_dir)
    dump_mlir = generates_dir / "temp.mlir"
    dispatch_dir = generates_dir / f"{model_name}_{device_spec}_dispatches"
    export_module_to_mlir_file(input_module, dump_mlir)
    dump_dispatches(
        dump_mlir,
        device,
        dispatch_dir,
        vulkan_target_triple,
        use_winograd=use_winograd,
    )

    # Tune each dispatch
    dtype = "f16" if args.precision == "fp16" else "f32"
    config_filename = f"{model_name}_{device_spec}_configs.json"

    for f_path in os.listdir(dispatch_dir):
        if not f_path.endswith(".mlir"):
            continue

        model_dir = os.path.join(dispatch_dir, f_path)

        tuner = SharkCodegenTuner(
            model_dir,
            device,
            "random",
            args.num_iters,
            args.tuned_config_dir,
            dtype,
            args.search_op,
            batch_size=1,
            config_filename=config_filename,
            use_dispatch=True,
            vulkan_target_triple=vulkan_target_triple,
        )
        tuner.tune()


if __name__ == "__main__":
    main()
