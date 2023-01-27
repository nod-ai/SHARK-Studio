import os
from shark.model_annotation import model_annotation, create_context
from shark.iree_utils._common import iree_target_map, run_cmd
from shark.shark_downloader import (
    download_model,
    download_public_file,
    WORKDIR,
)
from shark.parser import shark_args
from stable_args import args


device = (
    args.device if "://" not in args.device else args.device.split("://")[0]
)


# Download the model (Unet or VAE fp16) from shark_tank
def load_model_from_tank():
    from opt_params import get_params, version, variant

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
    config_bucket = "gs://shark_tank/sd_tuned/configs/"
    config_name = f"{args.annotation_model}_winograd_{device}.json"
    full_gs_url = config_bucket + config_name
    winograd_config_dir = f"{WORKDIR}configs/" + config_name
    print("Loading Winograd config file from ", winograd_config_dir)
    download_public_file(full_gs_url, winograd_config_dir, True)
    return winograd_config_dir


def load_lower_configs():
    from opt_params import version, variant

    config_bucket = "gs://shark_tank/sd_tuned/configs/"
    config_version = version
    if variant in ["anythingv3", "analogdiffusion"]:
        args.max_length = 77
        config_version = "v1_4"
    if args.annotation_model == "vae":
        args.max_length = 77
    config_name = f"{args.annotation_model}_{config_version}_{args.precision}_len{args.max_length}_{device}.json"
    full_gs_url = config_bucket + config_name
    lowering_config_dir = f"{WORKDIR}configs/" + config_name
    print("Loading lowering config file from ", lowering_config_dir)
    download_public_file(full_gs_url, lowering_config_dir, True)
    return lowering_config_dir


# Annotate the model with Winograd attribute on selected conv ops
def annotate_with_winograd(input_mlir, winograd_config_dir, model_name):
    if model_name.split("_")[-1] != "tuned":
        out_file_path = (
            f"{args.annotation_output}/{model_name}_tuned_torch.mlir"
        )
    else:
        out_file_path = f"{args.annotation_output}/{model_name}_torch.mlir"

    with create_context() as ctx:
        winograd_model = model_annotation(
            ctx,
            input_contents=input_mlir,
            config_path=winograd_config_dir,
            search_op="conv",
            winograd=True,
        )
        with open(out_file_path, "w") as f:
            f.write(str(winograd_model))
            f.close()
    return winograd_model, out_file_path


# For Unet annotate the model with tuned lowering configs
def annotate_with_lower_configs(
    input_mlir, lowering_config_dir, model_name, use_winograd
):
    if use_winograd:
        dump_after = "iree-linalg-ext-convert-conv2d-to-winograd"
    else:
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
    print("Applying tuned configs on", model_name)

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
    if model_name.split("_")[-1] != "tuned":
        out_file_path = (
            f"{args.annotation_output}/{model_name}_tuned_torch.mlir"
        )
    else:
        out_file_path = f"{args.annotation_output}/{model_name}_torch.mlir"
    with open(out_file_path, "w") as f:
        f.write(str(tuned_model))
        f.close()
    return tuned_model, out_file_path


def sd_model_annotation(mlir_model, model_name, model_from_tank=False):
    if args.annotation_model == "unet" and device == "vulkan":
        use_winograd = True
        winograd_config_dir = load_winograd_configs()
        winograd_model, model_path = annotate_with_winograd(
            mlir_model, winograd_config_dir, model_name
        )
        lowering_config_dir = load_lower_configs()
        tuned_model, output_path = annotate_with_lower_configs(
            model_path, lowering_config_dir, model_name, use_winograd
        )
    elif args.annotation_model == "vae" and device == "vulkan":
        use_winograd = True
        winograd_config_dir = load_winograd_configs()
        tuned_model, output_path = annotate_with_winograd(
            mlir_model, winograd_config_dir, model_name
        )
    else:
        use_winograd = False
        if model_from_tank:
            mlir_model = f"{WORKDIR}{model_name}_torch/{model_name}_torch.mlir"
        else:
            # Just use this function to convert bytecode to string
            orig_model, model_path = annotate_with_winograd(
                mlir_model, "", model_name
            )
            mlir_model = model_path
        lowering_config_dir = load_lower_configs()
        tuned_model, output_path = annotate_with_lower_configs(
            mlir_model, lowering_config_dir, model_name, use_winograd
        )
    print(f"Saved the annotated mlir in {output_path}.")
    return tuned_model, output_path


if __name__ == "__main__":
    mlir_model, model_name = load_model_from_tank()
    sd_model_annotation(mlir_model, model_name, model_from_tank=True)
