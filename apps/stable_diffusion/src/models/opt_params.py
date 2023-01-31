import sys
from transformers import CLIPTokenizer
from ..utils import models_db, args, get_shark_model


hf_model_variant_map = {
    "Linaqruf/anything-v3.0": ["anythingv3", "v2_1base"],
    "dreamlike-art/dreamlike-diffusion-1.0": ["dreamlike", "v2_1base"],
    "prompthero/openjourney": ["openjourney", "v2_1base"],
    "wavymulder/Analog-Diffusion": ["analogdiffusion", "v2_1base"],
    "stabilityai/stable-diffusion-2-1": ["stablediffusion", "v2_1"],
    "stabilityai/stable-diffusion-2-1-base": ["stablediffusion", "v2_1base"],
    "CompVis/stable-diffusion-v1-4": ["stablediffusion", "v1_4"],
}


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
            f"{bucket_key}/{model_key} is not present in the models database"
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
    variant, version = hf_model_variant_map[args.hf_model_id]
    # Tuned model is present only for `fp16` precision.
    is_tuned = "tuned" if args.use_tuned else "untuned"
    if "vulkan" not in args.device and args.use_tuned:
        bucket_key = f"{variant}/{is_tuned}/{args.device}"
        model_key = f"{variant}/{version}/unet/{args.precision}/length_{args.max_length}/{is_tuned}/{args.device}"
    else:
        bucket_key = f"{variant}/{is_tuned}"
        model_key = f"{variant}/{version}/unet/{args.precision}/length_{args.max_length}/{is_tuned}"

    bucket, model_name, iree_flags = get_params(
        bucket_key, model_key, "unet", is_tuned, args.precision
    )
    return get_shark_model(bucket, model_name, iree_flags)


def get_vae():
    variant, version = hf_model_variant_map[args.hf_model_id]
    # Tuned model is present only for `fp16` precision.
    is_tuned = "tuned" if args.use_tuned else "untuned"
    is_base = "/base" if args.use_base_vae else ""
    if "vulkan" not in args.device and args.use_tuned:
        bucket_key = f"{variant}/{is_tuned}/{args.device}"
        model_key = f"{variant}/{version}/vae/{args.precision}/length_77/{is_tuned}{is_base}/{args.device}"
    else:
        bucket_key = f"{variant}/{is_tuned}"
        model_key = f"{variant}/{version}/vae/{args.precision}/length_77/{is_tuned}{is_base}"

    bucket, model_name, iree_flags = get_params(
        bucket_key, model_key, "vae", is_tuned, args.precision
    )
    return get_shark_model(bucket, model_name, iree_flags)


def get_clip():
    variant, version = hf_model_variant_map[args.hf_model_id]
    bucket_key = f"{variant}/untuned"
    model_key = (
        f"{variant}/{version}/clip/fp32/length_{args.max_length}/untuned"
    )
    bucket, model_name, iree_flags = get_params(
        bucket_key, model_key, "clip", "untuned", "fp32"
    )
    return get_shark_model(bucket, model_name, iree_flags)


def get_tokenizer():
    tokenizer = CLIPTokenizer.from_pretrained(
        args.hf_model_id, subfolder="tokenizer"
    )
    return tokenizer
