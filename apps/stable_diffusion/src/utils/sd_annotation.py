import os
import io
from shark.model_annotation import model_annotation, create_context
from shark.iree_utils._common import iree_target_map, run_cmd
from shark.shark_downloader import (
    download_model,
    download_public_file,
    WORKDIR,
)
from shark.parser import shark_args
from apps.stable_diffusion.src.utils.stable_args import args


def get_device():
    device = (
        args.device
        if "://" not in args.device
        else args.device.split("://")[0]
    )
    return device


def get_device_args():
    device = get_device()
    device_spec_args = []
    if device == "cuda":
        from shark.iree_utils.gpu_utils import get_iree_gpu_args

        gpu_flags = get_iree_gpu_args()
        for flag in gpu_flags:
            device_spec_args.append(flag)
    elif device == "vulkan":
        device_spec_args.append(
            f"--iree-vulkan-target-triple={args.iree_vulkan_target_triple} "
        )
    return device, device_spec_args


# Download the model (Unet or VAE fp16) from shark_tank
def load_model_from_tank():
    from apps.stable_diffusion.src.models import (
        get_params,
        get_variant_version,
    )

    variant, version = get_variant_version(args.hf_model_id)

    shark_args.local_tank_cache = args.local_tank_cache
    bucket_key = f"{variant}/untuned"
    if args.annotation_model == "unet":
        model_key = f"{variant}/{version}/unet/{args.precision}/length_{args.max_length}/untuned"
    elif args.annotation_model == "vae":
        is_base = "/base" if args.use_base_vae else ""
        model_key = f"{variant}/{version}/vae/{args.precision}/length_77/untuned{is_base}"

    bucket, model_name, iree_flags = get_params(
        bucket_key, model_key, args.annotation_model, "untuned", args.precision
    )
    mlir_model, func_name, inputs, golden_out = download_model(
        model_name,
        tank_url=bucket,
        frontend="torch",
    )
    return mlir_model, model_name


# Download the tuned config files from shark_tank
def load_winograd_configs():
    device = get_device()
    config_bucket = "gs://shark_tank/sd_tuned/configs/"
    config_name = f"{args.annotation_model}_winograd_{device}.json"
    full_gs_url = config_bucket + config_name
    winograd_config_dir = os.path.join(WORKDIR, "configs", config_name)
    print("Loading Winograd config file from ", winograd_config_dir)
    download_public_file(full_gs_url, winograd_config_dir, True)
    return winograd_config_dir


def load_lower_configs(base_model_id=None):
    from apps.stable_diffusion.src.models import get_variant_version
    from apps.stable_diffusion.src.utils.utils import (
        fetch_and_update_base_model_id,
    )

    if not base_model_id:
        if args.ckpt_loc != "":
            base_model_id = fetch_and_update_base_model_id(args.ckpt_loc)
        else:
            base_model_id = fetch_and_update_base_model_id(args.hf_model_id)
            if base_model_id == "":
                base_model_id = args.hf_model_id

    variant, version = get_variant_version(base_model_id)

    if version == "inpaint_v1":
        version = "v1_4"
    elif version == "inpaint_v2":
        version = "v2_1base"

    config_bucket = "gs://shark_tank/sd_tuned_configs/"

    device, device_spec_args = get_device_args()
    spec = ""
    if device_spec_args:
        spec = device_spec_args[-1].split("=")[-1].strip()
        if device == "vulkan":
            spec = spec.split("-")[0]

    if args.annotation_model == "vae":
        if not spec or spec in ["rdna3", "sm_80"]:
            config_name = (
                f"{args.annotation_model}_{args.precision}_{device}.json"
            )
        else:
            config_name = f"{args.annotation_model}_{args.precision}_{device}_{spec}.json"
    else:
        if not spec or spec in ["rdna3", "sm_80"]:
            config_name = f"{args.annotation_model}_{version}_{args.precision}_{device}.json"
        else:
            config_name = f"{args.annotation_model}_{version}_{args.precision}_{device}_{spec}.json"

    full_gs_url = config_bucket + config_name
    lowering_config_dir = os.path.join(WORKDIR, "configs", config_name)
    print("Loading lowering config file from ", lowering_config_dir)
    download_public_file(full_gs_url, lowering_config_dir, True)
    return lowering_config_dir


# Annotate the model with Winograd attribute on selected conv ops
def annotate_with_winograd(input_mlir, winograd_config_dir, model_name):
    with create_context() as ctx:
        winograd_model = model_annotation(
            ctx,
            input_contents=input_mlir,
            config_path=winograd_config_dir,
            search_op="conv",
            winograd=True,
        )

    bytecode_stream = io.BytesIO()
    winograd_model.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    if args.save_annotation:
        if model_name.split("_")[-1] != "tuned":
            out_file_path = os.path.join(
                args.annotation_output, model_name + "_tuned_torch.mlir"
            )
        else:
            out_file_path = os.path.join(
                args.annotation_output, model_name + "_torch.mlir"
            )
        with open(out_file_path, "w") as f:
            f.write(str(winograd_model))
            f.close()

    return bytecode


def dump_after_mlir(input_mlir, use_winograd):
    import iree.compiler as ireec

    device, device_spec_args = get_device_args()
    if use_winograd:
        preprocess_flag = "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32},iree-linalg-ext-convert-conv2d-to-winograd))"
    else:
        preprocess_flag = "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32}))"

    dump_module = ireec.compile_str(
        input_mlir,
        target_backends=[iree_target_map(device)],
        extra_args=device_spec_args
        + [
            preprocess_flag,
            "--compile-to=preprocessing",
        ],
    )
    return dump_module


# For Unet annotate the model with tuned lowering configs
def annotate_with_lower_configs(
    input_mlir, lowering_config_dir, model_name, use_winograd
):
    # Dump IR after padding/img2col/winograd passes
    dump_module = dump_after_mlir(input_mlir, use_winograd)
    print("Applying tuned configs on", model_name)

    # Annotate the model with lowering configs in the config file
    with create_context() as ctx:
        tuned_model = model_annotation(
            ctx,
            input_contents=dump_module,
            config_path=lowering_config_dir,
            search_op="all",
        )

    bytecode_stream = io.BytesIO()
    tuned_model.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    if args.save_annotation:
        if model_name.split("_")[-1] != "tuned":
            out_file_path = (
                f"{args.annotation_output}/{model_name}_tuned_torch.mlir"
            )
        else:
            out_file_path = f"{args.annotation_output}/{model_name}_torch.mlir"
        with open(out_file_path, "w") as f:
            f.write(str(tuned_model))
            f.close()

    return bytecode


def sd_model_annotation(mlir_model, model_name, base_model_id=None):
    device = get_device()
    if args.annotation_model == "unet" and device == "vulkan":
        use_winograd = True
        winograd_config_dir = load_winograd_configs()
        winograd_model = annotate_with_winograd(
            mlir_model, winograd_config_dir, model_name
        )
        lowering_config_dir = load_lower_configs(base_model_id)
        tuned_model = annotate_with_lower_configs(
            winograd_model, lowering_config_dir, model_name, use_winograd
        )
    elif args.annotation_model == "vae" and device == "vulkan":
        use_winograd = True
        winograd_config_dir = load_winograd_configs()
        tuned_model = annotate_with_winograd(
            mlir_model, winograd_config_dir, model_name
        )
    else:
        use_winograd = False
        lowering_config_dir = load_lower_configs(base_model_id)
        tuned_model = annotate_with_lower_configs(
            mlir_model, lowering_config_dir, model_name, use_winograd
        )
    return tuned_model


if __name__ == "__main__":
    mlir_model, model_name = load_model_from_tank()
    sd_model_annotation(mlir_model, model_name)
