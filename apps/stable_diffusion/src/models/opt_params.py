import sys
from transformers import CLIPTokenizer
from apps.stable_diffusion.src.utils import (
    models_db,
    args,
    get_shark_model,
    get_opt_flags,
)


hf_model_variant_map = {
    "Linaqruf/anything-v3.0": ["anythingv3", "v1_4"],
    "dreamlike-art/dreamlike-diffusion-1.0": ["dreamlike", "v1_4"],
    "prompthero/openjourney": ["openjourney", "v1_4"],
    "wavymulder/Analog-Diffusion": ["analogdiffusion", "v1_4"],
    "stabilityai/stable-diffusion-2-1": ["stablediffusion", "v2_1base"],
    "stabilityai/stable-diffusion-2-1-base": ["stablediffusion", "v2_1base"],
    "CompVis/stable-diffusion-v1-4": ["stablediffusion", "v1_4"],
    "runwayml/stable-diffusion-inpainting": ["stablediffusion", "inpaint_v1"],
    "stabilityai/stable-diffusion-2-inpainting": [
        "stablediffusion",
        "inpaint_v2",
    ],
}


# TODO: Add the quantized model as a part model_db.json.
# This is currently in experimental phase.
def get_quantize_model():
    bucket_key = "gs://shark_tank/prashant_nod"
    model_key = "unet_int8"
    iree_flags = get_opt_flags("unet", precision="fp16")
    if args.height != 512 and args.width != 512 and args.max_length != 77:
        sys.exit(
            "The int8 quantized model currently requires the height and width to be 512, and max_length to be 77"
        )
    return bucket_key, model_key, iree_flags


def get_variant_version(hf_model_id):
    return hf_model_variant_map[hf_model_id]


def get_params(bucket_key, model_key, model, is_tuned, precision):
    try:
        bucket = models_db[0][bucket_key]
        model_name = models_db[1][model_key]
    except KeyError:
        raise Exception(
            f"{bucket_key}/{model_key} is not present in the models database"
        )
    iree_flags = get_opt_flags(model, precision="fp16")
    return bucket, model_name, iree_flags


def get_unet():
    variant, version = get_variant_version(args.hf_model_id)
    # Tuned model is present only for `fp16` precision.
    is_tuned = "tuned" if args.use_tuned else "untuned"

    # TODO: Get the quantize model from model_db.json
    if args.use_quantize == "int8":
        bk, mk, flags = get_quantize_model()
        return get_shark_model(bk, mk, flags)

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


def get_vae_encode():
    variant, version = get_variant_version(args.hf_model_id)
    # Tuned model is present only for `fp16` precision.
    is_tuned = "tuned" if args.use_tuned else "untuned"
    if "vulkan" not in args.device and args.use_tuned:
        bucket_key = f"{variant}/{is_tuned}/{args.device}"
        model_key = f"{variant}/{version}/vae_encode/{args.precision}/length_77/{is_tuned}/{args.device}"
    else:
        bucket_key = f"{variant}/{is_tuned}"
        model_key = f"{variant}/{version}/vae_encode/{args.precision}/length_77/{is_tuned}"

    bucket, model_name, iree_flags = get_params(
        bucket_key, model_key, "vae", is_tuned, args.precision
    )
    return get_shark_model(bucket, model_name, iree_flags)


def get_vae():
    variant, version = get_variant_version(args.hf_model_id)
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
    variant, version = get_variant_version(args.hf_model_id)
    bucket_key = f"{variant}/untuned"
    model_key = (
        f"{variant}/{version}/clip/fp32/length_{args.max_length}/untuned"
    )
    bucket, model_name, iree_flags = get_params(
        bucket_key, model_key, "clip", "untuned", "fp32"
    )
    return get_shark_model(bucket, model_name, iree_flags)


def get_tokenizer(subfolder="tokenizer"):
    tokenizer = CLIPTokenizer.from_pretrained(
        args.hf_model_id, subfolder=subfolder
    )
    return tokenizer
