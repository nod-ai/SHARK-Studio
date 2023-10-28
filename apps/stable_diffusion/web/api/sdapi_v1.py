import base64
import os
import pickle

from argparse import Namespace
from fastapi.exceptions import HTTPException
from io import BytesIO
from PIL import Image

from apps.stable_diffusion.src import args
from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    get_custom_model_files,
    get_custom_model_pathfile,
    predefined_models,
    predefined_paint_models,
    predefined_upscaler_models,
    scheduler_list,
)
from apps.stable_diffusion.web.ui.txt2img_ui import txt2img_inf
from apps.stable_diffusion.web.ui.img2img_ui import img2img_inf
from apps.stable_diffusion.web.ui.inpaint_ui import inpaint_inf
from apps.stable_diffusion.web.ui.outpaint_ui import outpaint_inf
from apps.stable_diffusion.web.ui.upscaler_ui import upscaler_inf

# Probably overly cautious, but try to ensure we only use the starting
# args in each api call, as the code does `args.<whatever> = <changed_value>`
# in lots of places and in testing it seemed to me these changes leaked
# into subsequent api calls.

# Roundtripping through pickle for deepcopy, there is probably a better way
frozen_args = Namespace(**(pickle.loads(pickle.dumps(vars(args)))))


# helper functions
def encode_pil_to_base64(images):
    encoded_imgs = []
    for image in images:
        with BytesIO() as output_bytes:
            if frozen_args.output_img_format.lower() == "png":
                image.save(output_bytes, format="PNG")

            elif frozen_args.output_img_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG")
            else:
                raise HTTPException(
                    status_code=500, detail="Invalid image format"
                )
            bytes_data = output_bytes.getvalue()
            encoded_imgs.append(base64.b64encode(bytes_data))
    return encoded_imgs


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";", 1)[1].split(",", 1)[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        print(err)
        raise HTTPException(status_code=400, detail="Invalid encoded image")


def get_predefined_models(custom_checkpoint_type):
    match custom_checkpoint_type:
        case "inpainting":
            return predefined_paint_models
        case "upscaler":
            return predefined_upscaler_models
        case _:
            return predefined_models


def get_model_from_request(
    request_json, checkpoint_type="", fallback_model=""
):
    # extract a model name from the request if available
    request_model = request_json.get(
        "hf_model_id",
        request_json.get("override_settings", {}).get(
            "sd_model_checkpoint", None
        ),
    )

    # if the request didn't specify a model try the command line args
    result = request_model or frozen_args.ckpt_loc or frozen_args.hf_model_id

    # make sure whatever we have is a valid model for the checkpoint type
    if result in get_custom_model_files(
        custom_checkpoint_type=checkpoint_type
    ) + get_predefined_models(checkpoint_type):
        return result
    # if not return what was specified as the fallback
    else:
        return fallback_model


def get_scheduler_from_request(request_json, fallback_scheduler=None):
    if request_json.get("sampler_name", None) in scheduler_list:
        result = request_json["sampler_name"]
    else:
        result = frozen_args.scheduler

    if result == "SharkEulerDiscrete" and fallback_scheduler:
        return fallback_scheduler
    else:
        return result


def bad_request_for_missing(input_data, required: list):
    missing = [key for key in required if key not in input_data.keys()]
    if len(missing) > 0:
        raise HTTPException(
            status_code=400, detail=f"Missing required parameters: {missing}"
        )


# Rest API: /sdapi/v1/sd-models (lists available models)
def sd_models_api():
    return [
        {
            "title": model_file,
            "model_name": model_file,
            "hash": None,
            "sha256": None,
            "filename": get_custom_model_pathfile(model_file),
            "config": None,
        }
        for model_file in get_custom_model_files()
    ] + [
        {
            "title": model,
            "model_name": model,
            "hash": None,
            "sha256": None,
            "filename": None,
            "config": None,
        }
        for model in predefined_models
    ]


# Rest API: /sdapi/v1/options (lists application level options)
def options_api():
    # This is mostly just enough to support what Koboldcpp wants
    return {
        "samples_save": True,
        "samples_format": frozen_args.output_img_format,
        "sd_model_checkpoint": os.path.basename(frozen_args.ckpt_loc)
        if frozen_args.ckpt_loc
        else frozen_args.hf_model_id,
    }


# Rest API: /sdapi/v1/txt2img (Text to image)
def txt2img_api(
    InputData: dict,
):
    bad_request_for_missing(InputData, ["prompt", "negative_prompt"])

    model_id = get_model_from_request(
        InputData, fallback_model="stabilityai/stable-diffusion-2-1-base"
    )
    scheduler = get_scheduler_from_request(
        InputData, "DEISMultistep" if frozen_args.use_hiresfix else None
    )

    print(
        f'Prompt: {InputData["prompt"]}, '
        f'Negative Prompt: {InputData["negative_prompt"]}, '
        f'Seed: {InputData["seed"] or -1},'
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}. "
    )

    res = txt2img_inf(
        InputData["prompt"],
        InputData["negative_prompt"],
        InputData.get("height", frozen_args.height),
        InputData.get("width", frozen_args.width),
        InputData.get("steps", frozen_args.steps),
        InputData.get("cfg_scale", frozen_args.guidance_scale),
        InputData.get("seed", frozen_args.seed),
        batch_count=1,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=available_devices[0],
        max_length=frozen_args.max_length,
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights="None",
        lora_hf_id="",
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
        use_hiresfix=frozen_args.use_hiresfix,
        hiresfix_height=frozen_args.hiresfix_height,
        hiresfix_width=frozen_args.hiresfix_width,
        hiresfix_strength=frozen_args.hiresfix_strength,
        resample_type=frozen_args.resample_type,
    )

    # Convert Generator to Subscriptable
    res = next(res)

    return {
        "images": encode_pil_to_base64(res[0]),
        "parameters": {},
        "info": res[1],
    }


# Rest API: /sdapi/v1/txt2img (Image to image)
def img2img_api(
    InputData: dict,
):
    bad_request_for_missing(
        InputData, ["prompt", "negative_prompt", "init_images"]
    )

    model_id = get_model_from_request(
        InputData,
        fallback_model="stabilityai/stable-diffusion-2-1-base",
    )
    init_image = decode_base64_to_image(InputData["init_images"][0])
    scheduler = get_scheduler_from_request(InputData, "EulerDiscrete")

    print(
        f'Prompt: {InputData["prompt"]}, '
        f'Negative Prompt: {InputData["negative_prompt"]}, '
        f'Seed: {InputData["seed"] or -1}, '
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}."
    )

    res = img2img_inf(
        InputData["prompt"],
        InputData["negative_prompt"],
        init_image,
        InputData.get("height", frozen_args.height),
        InputData.get("width", frozen_args.width),
        InputData.get("steps", frozen_args.steps),
        InputData.get("denoising_strength", frozen_args.strength),
        InputData.get("cfg_scale", frozen_args.guidance_scale),
        InputData.get("seed", frozen_args.seed),
        batch_count=1,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=available_devices[0],
        max_length=frozen_args.max_length,
        use_stencil=InputData.get("use_stencil", frozen_args.use_stencil),
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights="None",
        lora_hf_id="",
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
        resample_type=frozen_args.resample_type,
    )

    # Converts generator type to subscriptable
    res = next(res)

    return {
        "images": encode_pil_to_base64(res[0]),
        "parameters": {},
        "info": res[1],
    }


# Rest API: /sdapi/v1/inpaint (Inpainting)
def inpaint_api(
    InputData: dict,
):
    bad_request_for_missing(
        InputData,
        [
            "prompt",
            "negative_prompt",
            "image",
            "mask",
            "is_full_res",
            "full_res_padding",
        ],
    )

    model_id = get_model_from_request(
        InputData,
        checkpoint_type="inpainting",
        fallback_model="stabilityai/stable-diffusion-2-inpainting",
    )
    scheduler = get_scheduler_from_request(InputData, "EulerDiscrete")
    init_image = decode_base64_to_image(InputData["image"])
    mask = decode_base64_to_image(InputData["mask"])

    print(
        f'Prompt: {InputData["prompt"]}, '
        f'Negative Prompt: {InputData["negative_prompt"]}, '
        f'Seed: {InputData.get("seed", frozen_args.seed)}, '
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}."
    )

    res = inpaint_inf(
        InputData["prompt"],
        InputData["negative_prompt"],
        {"image": init_image, "mask": mask},
        InputData.get("height", frozen_args.height),
        InputData.get("width", frozen_args.width),
        InputData["is_full_res"],
        InputData["full_res_padding"],
        InputData.get("steps", frozen_args.steps),
        InputData.get("cfg_scale", frozen_args.guidance_scale),
        InputData.get("seed", frozen_args.seed),
        batch_count=1,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=available_devices[0],
        max_length=frozen_args.max_length,
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights="None",
        lora_hf_id="",
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
    )

    # Converts generator type to subscriptable
    res = next(res)

    return {
        "images": encode_pil_to_base64(res[0]),
        "parameters": {},
        "info": res[1],
    }


# Rest API: /sdapi/v1/outpaint (Outpainting)
def outpaint_api(
    InputData: dict,
):
    bad_request_for_missing(
        InputData, ["prompt", "negative_prompt", "init_images", "directions"]
    )

    model_id = get_model_from_request(
        InputData,
        checkpoint_type="inpainting",
        fallback_model="stabilityai/stable-diffusion-2-inpainting",
    )
    # Tested as working fallback CPU scheduler with the fallback model, many
    # other schedulers aren't currently working either here or in the webui
    scheduler = get_scheduler_from_request(InputData, "DDIM")
    init_image = decode_base64_to_image(InputData["init_images"][0])

    print(
        f'Prompt: {InputData["prompt"]}, '
        f'Negative Prompt: {InputData["negative_prompt"]}, '
        f'Seed: {InputData["seed"]}'
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}."
    )

    res = outpaint_inf(
        InputData["prompt"],
        InputData["negative_prompt"],
        init_image,
        InputData.get("pixels", frozen_args.pixels),
        InputData.get("mask_blur", frozen_args.mask_blur),
        InputData["directions"],  # TODO: 4 args to become 1
        InputData.get("noise_q", frozen_args.noise_q),
        InputData.get("color_variation", frozen_args.color_variation),
        InputData.get("height", frozen_args.height),
        InputData.get("width", frozen_args.width),
        InputData.get("steps", frozen_args.steps),
        InputData.get("cfg_scale", frozen_args.guidance_scale),
        InputData.get("seed", frozen_args.seed),
        batch_count=1,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=available_devices[0],
        max_length=frozen_args.max_length,
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights="None",
        lora_hf_id="",
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
    )

    # Convert Generator to Subscriptable
    res = next(res)

    return {
        "images": encode_pil_to_base64(res[0]),
        "parameters": {},
        "info": res[1],
    }


# Rest API: /sdapi/v1/upscaler (Upscaling)
def upscaler_api(
    InputData: dict,
):
    bad_request_for_missing(
        InputData, ["prompt", "negative_prompt", "init_images"]
    )

    model_id = get_model_from_request(
        InputData,
        checkpoint_type="upscaler",
        fallback_model="stabilityai/stable-diffusion-x4-upscaler",
    )
    init_image = decode_base64_to_image(InputData["init_images"][0])
    scheduler = get_scheduler_from_request(InputData, "EulerDiscrete")

    print(
        f'Prompt: {InputData["prompt"]}, '
        f'Negative Prompt: {InputData["negative_prompt"]}, '
        f'Seed: {InputData["seed"]}, '
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}."
    )

    res = upscaler_inf(
        InputData["prompt"],
        InputData["negative_prompt"],
        init_image,
        InputData.get("height", frozen_args.height),
        InputData.get("width", frozen_args.width),
        InputData.get("steps", frozen_args.steps),
        InputData.get("noise_level", frozen_args.noise_level),
        InputData.get("cfg_scale", frozen_args.guidance_scale),
        InputData.get("seed", frozen_args.seed),
        batch_count=1,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=available_devices[0],
        max_length=frozen_args.max_length,
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights="None",
        lora_hf_id="",
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
    )
    # Converts generator type to subscriptable
    res = next(res)

    return {
        "images": encode_pil_to_base64(res[0]),
        "parameters": {},
        "info": res[1],
    }
