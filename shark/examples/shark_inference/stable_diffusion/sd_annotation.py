import os
from shark.model_annotation import model_annotation, create_context
from shark.iree_utils._common import run_cmd, iree_target_map
from shark.shark_downloader import (
    download_model,
    download_public_file,
    WORKDIR,
)
from shark.parser import shark_args
from stable_args import args
from opt_params import get_params
from utils import set_init_device_flags


set_init_device_flags()
device = (
    args.device if "://" not in args.device else args.device.split("://")[0]
)

# Downloads the model (Unet or VAE fp16) from shark_tank
shark_args.local_tank_cache = args.local_tank_cache
bucket_key = f"{args.variant}/untuned"
if args.annotation_model == "unet":
    model_key = f"{args.variant}/{args.version}/unet/{args.precision}/length_{args.max_length}/untuned"
elif args.annotation_model == "vae":
    is_base = "/base" if args.use_base_vae else ""
    model_key = f"{args.variant}/{args.version}/vae/{args.precision}/length_77/untuned{is_base}"

bucket, model_name, iree_flags = get_params(
    bucket_key, model_key, args.annotation_model, "untuned", args.precision
)
mlir_model, func_name, inputs, golden_out = download_model(
    model_name,
    tank_url=bucket,
    frontend="torch",
)

# Downloads the tuned config files from shark_tank
config_bucket = "gs://shark_tank/sd_tuned/configs/"
if args.use_winograd:
    config_name = f"{args.annotation_model}_winograd_{device}.json"
    full_gs_url = config_bucket + config_name
    winograd_config_dir = f"{WORKDIR}configs/" + config_name
    download_public_file(full_gs_url, winograd_config_dir, True)

if args.annotation_model == "unet" or device == "cuda":
    if args.variant in ["anythingv3", "analogdiffusion"] or args.annotation_model == "vae":
        args.max_length = 77
    config_name = f"{args.annotation_model}_{args.version}_{args.precision}_len{args.max_length}_{device}.json"
    full_gs_url = config_bucket + config_name
    lowering_config_dir = f"{WORKDIR}configs/" + config_name
    download_public_file(full_gs_url, lowering_config_dir, True)

# Annotate the model with Winograd attribute on selected conv ops
if args.use_winograd:
    with create_context() as ctx:
        winograd_model = model_annotation(
            ctx,
            input_contents=mlir_model,
            config_path=winograd_config_dir,
            search_op="conv",
            winograd=args.use_winograd,
        )
        with open(
            f"{args.annotation_output}/{model_name}_tuned_torch.mlir", "w"
        ) as f:
            f.write(str(winograd_model))

# For Unet annotate the model with tuned lowering configs
if args.annotation_model == "unet" or device == "cuda":
    if args.use_winograd:
        input_mlir = f"{args.annotation_output}/{model_name}_tuned_torch.mlir"
        dump_after = "iree-linalg-ext-convert-conv2d-to-winograd"
    else:
        input_mlir = f"{WORKDIR}{model_name}_torch/{model_name}_torch.mlir"
        dump_after = "iree-flow-pad-linalg-ops"

    # Dump IR after padding/img2col/winograd passes
    device_spec_args = ""
    if device == "cuda":
        from shark.iree_utils.gpu_utils import get_iree_gpu_args

        gpu_flags = get_iree_gpu_args()
        for flag in gpu_flags:
            device_spec_args += flag + " "
    elif device == "vulkan":
        device_spec_args = (
            f"--iree-vulkan-target-triple={args.iree_vulkan_target_triple} "
        )
    run_cmd(
        f"iree-compile {input_mlir} "
        "--iree-input-type=tm_tensor "
        f"--iree-hal-target-backends={iree_target_map(device)} "
        f"{device_spec_args}"
        "--iree-stream-resource-index-bits=64 "
        "--iree-vm-target-index-bits=64 "
        "--iree-flow-enable-padding-linalg-ops "
        "--iree-flow-linalg-ops-padding-size=32 "
        "--iree-flow-enable-conv-img2col-transform "
        f"--mlir-print-ir-after={dump_after} "
        "--compile-to=flow "
        f"2>{args.annotation_output}/dump_after_winograd.mlir "
    )

    # Annotate the model with lowering configs in the config file
    with create_context() as ctx:
        tuned_model = model_annotation(
            ctx,
            input_contents=f"{args.annotation_output}/dump_after_winograd.mlir",
            config_path=lowering_config_dir,
            search_op="all",
        )

    # Remove the intermediate mlir and save the final annotated model
    os.remove(f"{args.annotation_output}/dump_after_winograd.mlir")
    output_path = f"{args.annotation_output}/{model_name}_tuned_torch.mlir"
    with open(output_path, "w") as f:
        f.write(str(tuned_model))
    print(f"Saved the annotated mlir in {output_path}.")
