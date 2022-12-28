import sys
from model_wrappers import (
    get_base_vae_mlir,
    get_vae_mlir,
    get_unet_mlir,
    get_clip_mlir,
)
from resources import models_db
from stable_args import args
from utils import (
    get_shark_model,
    get_vmfb,
    update_checkpoint,
)
from shark.shark_inference import SharkInference

BATCH_SIZE = len(args.prompts)
if BATCH_SIZE != 1:
    sys.exit("Only batch size 1 is supported.")


def get_params(bucket_key, model_key, model, is_tuned, precision):
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )

    # Disable bindings fusion to work with moltenVK.
    if sys.platform == "darwin":
        iree_flags.append("-iree-stream-fuse-binding=false")

    try:
        bucket = models_db[0][bucket_key]
        model_name = models_db[1][model_key]
        iree_flags += models_db[2][model][is_tuned][precision][
            "default_compilation_flags"
        ]
    except KeyError:
        raise Exception(
            f"{bucket}/{model_key} is not present in the models database"
        )

    if (
        "specified_compilation_flags"
        in models_db[2][model][is_tuned][precision]
    ):
        device = (
            args.device
            if "://" not in args.device
            else args.device.split("://")[0]
        )
        if (
            device
            not in models_db[2][model][is_tuned][precision][
                "specified_compilation_flags"
            ]
        ):
            device = "default_device"
        iree_flags += models_db[2][model][is_tuned][precision][
            "specified_compilation_flags"
        ][device]

    return bucket, model_name, iree_flags


def get_unet():
    # Tuned model is present only for `fp16` precision.
    is_tuned = "tuned" if args.use_tuned else "untuned"
    bucket_key = f"{args.variant}/{is_tuned}"
    model_key = f"{args.variant}/{args.version}/unet/{args.precision}/length_{args.max_length}/{is_tuned}"
    bucket, model_name, iree_flags = get_params(
        bucket_key, model_key, "unet", is_tuned, args.precision
    )
    if not args.use_tuned and args.import_mlir:
        return get_unet_mlir(model_name, iree_flags)
    unet_model = get_shark_model(bucket, model_name, iree_flags)
    # TODO: Currently we only support untuned checkpoint updates.
    if args.unet_checkpoint != "" and not args.use_tuned:
        unet_ts = get_unet_mlir(model_name, iree_flags, get_ts_graph=True)
        unet_model = update_checkpoint(unet_model, unet_ts)
    return unet_model


def get_vae():
    # Tuned model is present only for `fp16` precision.
    is_tuned = "tuned" if args.use_tuned else "untuned"
    is_base = "/base" if args.use_base_vae else ""
    bucket_key = f"{args.variant}/{is_tuned}"
    model_key = f"{args.variant}/{args.version}/vae/{args.precision}/length_77/{is_tuned}{is_base}"
    bucket, model_name, iree_flags = get_params(
        bucket_key, model_key, "vae", is_tuned, args.precision
    )
    if not args.use_tuned and args.import_mlir:
        if args.use_base_vae:
            return get_base_vae_mlir(model_name, iree_flags)
        return get_vae_mlir(model_name, iree_flags)
    vae_model = get_shark_model(bucket, model_name, iree_flags)
    # TODO: Currently we only support untuned checkpoint updates.
    if args.vae_checkpoint != "" and not args.use_tuned:
        vae_ts = None
        if args.use_base_vae:
            vae_ts = get_base_vae_mlir(model_name, iree_flags, get_ts_graph=True)
        else:
            vae_ts = get_vae_mlir(model_name, iree_flags, get_ts_graph=True)
        vae_model = update_checkpoint(vae_model, vae_ts)
    return vae_model


def get_clip():
    bucket_key = f"{args.variant}/untuned"
    model_key = f"{args.variant}/{args.version}/clip/fp32/length_{args.max_length}/untuned"
    bucket, model_name, iree_flags = get_params(
        bucket_key, model_key, "clip", "untuned", "fp32"
    )
    if args.import_mlir:
        return get_clip_mlir(model_name, iree_flags)
    clip_model = get_shark_model(bucket, model_name, iree_flags)
    # TODO: Currently we only support untuned checkpoint updates.
    if args.clip_checkpoint != "" and not args.use_tuned:
        clip_ts = get_clip_mlir(model_name, iree_flags, get_ts_graph=True)
        clip_model = update_checkpoint(clip_model, clip_ts)
    return clip_model
