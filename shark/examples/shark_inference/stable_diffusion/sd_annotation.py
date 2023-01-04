import os
from shark.model_annotation import model_annotation, create_context
from shark.iree_utils._common import run_cmd, iree_target_map
from shark.shark_downloader import download_model
from shark.parser import shark_args
from stable_args import args
from opt_params import get_params
from utils import set_init_device_flags


# Downloads the model (Unet or VAE fp16) from shark_tank
set_init_device_flags()
shark_args.local_tank_cache = args.local_tank_cache
bucket_key = f"{args.variant}/untuned"
winograd_opt = 0
if args.model == "unet":
    winograd_opt = 1
    model_key = f"{args.variant}/{args.version}/unet/{args.precision}/length_{args.max_length}/untuned"
elif args.model == "vae":
    winograd_opt = 2
    is_base = "/base" if args.use_base_vae else ""
    model_key = f"{args.variant}/{args.version}/vae/{args.precision}/length_77/untuned{is_base}"

bucket, model_name, iree_flags = get_params(
    bucket_key, model_key, args.model, "untuned", args.precision
)
mlir_model, func_name, inputs, golden_out = download_model(
    model_name,
    tank_url=bucket,
    frontend="torch",
)

# Annotate the model with Winograd attribute on selected conv ops
with create_context() as ctx:
    winograd_model = model_annotation(
        ctx,
        input_contents=mlir_model,
        config_path=args.config_path,
        search_op="conv",
        winograd=winograd_opt,
    )
    with open(f"{args.output_dir}/{model_name}_tuned_torch.mlir", "w") as f:
        f.write(str(winograd_model))

if args.model == "unet":
    # Dump IR after padding/img2col/winograd passes
    run_cmd(
        "iree-compile "
        f"{args.output_dir}/{model_name}_tuned_torch.mlir "
        "--iree-input-type=tm_tensor "
        f"--iree-hal-target-backends={iree_target_map(args.device)} "
        f"--iree-vulkan-target-triple={args.iree_vulkan_target_triple} "
        "--iree-stream-resource-index-bits=64 "
        "--iree-vm-target-index-bits=64 "
        "--iree-flow-enable-padding-linalg-ops "
        "--iree-flow-linalg-ops-padding-size=32 "
        "--iree-flow-enable-conv-img2col-transform "
        "--mlir-print-ir-after=iree-linalg-ext-convert-conv2d-to-winograd "
        f"-o {args.output_dir}/dump_after_winograd.vmfb "
        f"2>{args.output_dir}/dump_after_winograd.mlir "
    )

    # Annotate the model with lowering configs in the config file
    with create_context() as ctx:
        tuned_model = model_annotation(
            ctx,
            input_contents=f"{args.output_dir}/dump_after_winograd.mlir",
            config_path=args.config_path,
            search_op="all",
        )

    # Remove the intermediate generates and save the final annotated model
    os.remove(f"{args.output_dir}/dump_after_winograd.vmfb")
    os.remove(f"{args.output_dir}/dump_after_winograd.mlir")
    output_path = f"{args.output_dir}/{model_name}_tuned_torch.mlir"
    with open(output_path, "w") as f:
        f.write(str(tuned_model))
    print(f"Saved the annotated mlir in {output_path}.")
