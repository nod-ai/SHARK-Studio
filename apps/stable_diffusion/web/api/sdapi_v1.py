import os

from collections import defaultdict
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel, Field

from apps.stable_diffusion.web.api.utils import (
    frozen_args,
    sampler_aliases,
    encode_pil_to_base64,
    decode_base64_to_image,
    get_model_from_request,
    get_scheduler_from_request,
    get_lora_params,
    get_device,
    GenerationInputData,
    GenerationResponseData,
)

from apps.stable_diffusion.web.ui.utils import (
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

sdapi = FastAPI()


# Rest API: /sdapi/v1/sd-models (lists available models)
class AppParam(str, Enum):
    txt2img = "txt2img"
    img2img = "img2img"
    inpaint = "inpaint"
    outpaint = "outpaint"
    upscaler = "upscaler"


@sdapi.get(
    "/v1/sd-models",
    summary="lists available models",
    description=(
        "This is all the models that this server currently knows about.\n "
        "Models listed may still have a compilation and build pending that "
        "will be triggered the first time they are used."
    ),
)
def sd_models_api(app: AppParam = frozen_args.app):
    match app:
        case "inpaint" | "outpaint":
            checkpoint_type = "inpainting"
            predefined = predefined_paint_models
        case "upscaler":
            checkpoint_type = "upscaler"
            predefined = predefined_upscaler_models
        case _:
            checkpoint_type = ""
            predefined = predefined_models

    return [
        {
            "title": model_file,
            "model_name": model_file,
            "hash": None,
            "sha256": None,
            "filename": get_custom_model_pathfile(model_file),
            "config": None,
        }
        for model_file in get_custom_model_files(
            custom_checkpoint_type=checkpoint_type
        )
    ] + [
        {
            "title": model,
            "model_name": model,
            "hash": None,
            "sha256": None,
            "filename": None,
            "config": None,
        }
        for model in predefined
    ]


# Rest API: /sdapi/v1/samplers (lists schedulers)
@sdapi.get(
    "/v1/samplers",
    summary="lists available schedulers/samplers",
    description=(
        "These are all the Schedulers defined and available. Not "
        "every scheduler is compatible with all apis. Aliases are "
        "equivalent samplers in A1111 if they are known."
    ),
)
def sd_samplers_api():
    reverse_sampler_aliases = defaultdict(list)
    for key, value in sampler_aliases.items():
        reverse_sampler_aliases[value].append(key)

    return (
        {
            "name": scheduler,
            "aliases": reverse_sampler_aliases.get(scheduler, []),
            "options": {},
        }
        for scheduler in scheduler_list
    )


# Rest API: /sdapi/v1/options (lists application level options)
@sdapi.get(
    "/v1/options",
    summary="lists current settings of application level options",
    description=(
        "A subset of the command line arguments set at startup renamed "
        "to correspond to the A1111 naming. Only a small subset of A1111 "
        "options are returned."
    ),
)
def options_api():
    # This is mostly just enough to support what Koboldcpp wants, with a
    # few other things that seemed obvious
    return {
        "samples_save": True,
        "samples_format": frozen_args.output_img_format,
        "sd_model_checkpoint": os.path.basename(frozen_args.ckpt_loc)
        if frozen_args.ckpt_loc
        else frozen_args.hf_model_id,
        "sd_lora": frozen_args.use_lora,
        "sd_vae": frozen_args.custom_vae or "Automatic",
        "enable_pnginfo": frozen_args.write_metadata_to_png,
    }


# Rest API: /sdapi/v1/cmd-flags (lists command line argument settings)
@sdapi.get(
    "/v1/cmd-flags",
    summary="lists the command line arguments value that were set on startup.",
)
def cmd_flags_api():
    return vars(frozen_args)


# Rest API: /sdapi/v1/txt2img (Text to image)
class ModelOverrideSettings(BaseModel):
    sd_model_checkpoint: str = get_model_from_request(
        fallback_model="stabilityai/stable-diffusion-2-1-base"
    )


class Txt2ImgInputData(GenerationInputData):
    enable_hr: bool = frozen_args.use_hiresfix
    hr_resize_y: int = Field(
        default=frozen_args.hiresfix_height, ge=128, le=768, multiple_of=8
    )
    hr_resize_x: int = Field(
        default=frozen_args.hiresfix_width, ge=128, le=768, multiple_of=8
    )
    override_settings: ModelOverrideSettings = None


@sdapi.post(
    "/v1/txt2img",
    summary="Does text to image generation",
    response_model=GenerationResponseData,
)
def txt2img_api(InputData: Txt2ImgInputData):
    # bad_request_for_missing(InputData, ["prompt", "negative_prompt"])

    model_id = get_model_from_request(
        InputData,
        fallback_model="stabilityai/stable-diffusion-2-1-base",
    )
    scheduler = get_scheduler_from_request(
        InputData, "txt2img_hires" if InputData.enable_hr else "txt2img"
    )
    (lora_weights, lora_hf_id) = get_lora_params(frozen_args.use_lora)

    print(
        f"Prompt: {InputData.prompt}, "
        f"Negative Prompt: {InputData.negative_prompt}, "
        f"Seed: {InputData.seed},"
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}. "
    )

    res = txt2img_inf(
        InputData.prompt,
        InputData.negative_prompt,
        InputData.height,
        InputData.width,
        InputData.steps,
        InputData.cfg_scale,
        InputData.seed,
        batch_count=InputData.n_iter,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=get_device(frozen_args.device),
        max_length=frozen_args.max_length,
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights=lora_weights,
        lora_hf_id=lora_hf_id,
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
        use_hiresfix=InputData.enable_hr,
        hiresfix_height=InputData.hr_resize_y,
        hiresfix_width=InputData.hr_resize_x,
        hiresfix_strength=frozen_args.hiresfix_strength,
        resample_type=frozen_args.resample_type,
    )

    # Since we're not streaming we just want the last generator result
    for items_so_far in res:
        items = items_so_far

    return {
        "images": encode_pil_to_base64(items[0]),
        "parameters": {},
        "info": items[1],
    }


# Rest API: /sdapi/v1/img2img (Image to image)
class StencilParam(str, Enum):
    canny = "canny"
    openpose = "openpose"
    scribble = "scribble"


class Img2ImgInputData(GenerationInputData):
    init_images: list[str]
    denoising_strength: float = frozen_args.strength
    use_stencil: StencilParam = frozen_args.use_stencil
    override_settings: ModelOverrideSettings = None


@sdapi.post(
    "/v1/img2img",
    summary="Does image to image generation",
    response_model=GenerationResponseData,
)
def img2img_api(
    InputData: Img2ImgInputData,
):
    model_id = get_model_from_request(
        InputData,
        fallback_model="stabilityai/stable-diffusion-2-1-base",
    )
    scheduler = get_scheduler_from_request(InputData, "img2img")
    (lora_weights, lora_hf_id) = get_lora_params(frozen_args.use_lora)

    init_image = decode_base64_to_image(InputData.init_images[0])

    print(
        f"Prompt: {InputData.prompt}, "
        f"Negative Prompt: {InputData.negative_prompt}, "
        f"Seed: {InputData.seed}, "
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}."
    )

    res = img2img_inf(
        InputData.prompt,
        InputData.negative_prompt,
        init_image,
        InputData.height,
        InputData.width,
        InputData.steps,
        InputData.denoising_strength,
        InputData.cfg_scale,
        InputData.seed,
        batch_count=InputData.n_iter,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=get_device(frozen_args.device),
        max_length=frozen_args.max_length,
        use_stencil=InputData.use_stencil,
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights=lora_weights,
        lora_hf_id=lora_hf_id,
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
        resample_type=frozen_args.resample_type,
    )

    # Since we're not streaming we just want the last generator result
    for items_so_far in res:
        items = items_so_far

    return {
        "images": encode_pil_to_base64(items[0]),
        "parameters": {},
        "info": items[1],
    }


# Rest API: /sdapi/v1/inpaint (Inpainting)
class PaintModelOverideSettings(BaseModel):
    sd_model_checkpoint: str = get_model_from_request(
        checkpoint_type="inpainting",
        fallback_model="stabilityai/stable-diffusion-2-inpainting",
    )


class InpaintInputData(GenerationInputData):
    image: str = Field(description="Base64 encoded input image")
    mask: str = Field(description="Base64 encoded mask image")
    is_full_res: bool = False  # Is this setting backwards in the UI?
    full_res_padding: int = Field(default=32, ge=0, le=256, multiple_of=4)
    denoising_strength: float = frozen_args.strength
    use_stencil: StencilParam = frozen_args.use_stencil
    override_settings: PaintModelOverideSettings = None


@sdapi.post(
    "/v1/inpaint",
    summary="Does inpainting generation on an image",
    response_model=GenerationResponseData,
)
def inpaint_api(
    InputData: InpaintInputData,
):
    model_id = get_model_from_request(
        InputData,
        checkpoint_type="inpainting",
        fallback_model="stabilityai/stable-diffusion-2-inpainting",
    )
    scheduler = get_scheduler_from_request(InputData, "inpaint")
    (lora_weights, lora_hf_id) = get_lora_params(frozen_args.use_lora)

    init_image = decode_base64_to_image(InputData.image)
    mask = decode_base64_to_image(InputData.mask)

    print(
        f"Prompt: {InputData.prompt}, "
        f'Negative Prompt: {InputData.negative_prompt}", '
        f'Seed: {InputData.seed}", '
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}."
    )

    res = inpaint_inf(
        InputData.prompt,
        InputData.negative_prompt,
        {"image": init_image, "mask": mask},
        InputData.height,
        InputData.width,
        InputData.is_full_res,
        InputData.full_res_padding,
        InputData.steps,
        InputData.cfg_scale,
        InputData.seed,
        batch_count=InputData.n_iter,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=get_device(frozen_args.device),
        max_length=frozen_args.max_length,
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights=lora_weights,
        lora_hf_id=lora_hf_id,
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
    )

    # Since we're not streaming we just want the last generator result
    for items_so_far in res:
        items = items_so_far

    return {
        "images": encode_pil_to_base64(items[0]),
        "parameters": {},
        "info": items[1],
    }


# Rest API: /sdapi/v1/outpaint (Outpainting)
class DirectionParam(str, Enum):
    left = "left"
    right = "right"
    up = "up"
    down = "down"


class OutpaintInputData(GenerationInputData):
    init_images: list[str]
    pixels: int = Field(
        default=frozen_args.pixels, ge=8, le=256, multiple_of=8
    )
    mask_blur: int = Field(default=frozen_args.mask_blur, ge=0, le=64)
    directions: set[DirectionParam] = [
        direction
        for direction in ["left", "right", "up", "down"]
        if vars(frozen_args)[direction]
    ]
    noise_q: float = frozen_args.noise_q
    color_variation: float = frozen_args.color_variation
    override_settings: PaintModelOverideSettings = None


@sdapi.post(
    "/v1/outpaint",
    summary="Does outpainting generation on an image",
    response_model=GenerationResponseData,
)
def outpaint_api(
    InputData: OutpaintInputData,
):
    model_id = get_model_from_request(
        InputData,
        checkpoint_type="inpainting",
        fallback_model="stabilityai/stable-diffusion-2-inpainting",
    )
    scheduler = get_scheduler_from_request(InputData, "outpaint")
    (lora_weights, lora_hf_id) = get_lora_params(frozen_args.use_lora)

    init_image = decode_base64_to_image(InputData.init_images[0])

    print(
        f"Prompt: {InputData.prompt}, "
        f"Negative Prompt: {InputData.negative_prompt}, "
        f"Seed: {InputData.seed}, "
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}."
    )

    res = outpaint_inf(
        InputData.prompt,
        InputData.negative_prompt,
        init_image,
        InputData.pixels,
        InputData.mask_blur,
        InputData.directions,
        InputData.noise_q,
        InputData.color_variation,
        InputData.height,
        InputData.width,
        InputData.steps,
        InputData.cfg_scale,
        InputData.seed,
        batch_count=InputData.n_iter,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=get_device(frozen_args.device),
        max_length=frozen_args.max_length,
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights=lora_weights,
        lora_hf_id=lora_hf_id,
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
    )

    # Since we're not streaming we just want the last generator result
    for items_so_far in res:
        items = items_so_far

    return {
        "images": encode_pil_to_base64(items[0]),
        "parameters": {},
        "info": items[1],
    }


# Rest API: /sdapi/v1/upscaler (Upscaling)
class UpscalerModelOverideSettings(BaseModel):
    sd_model_checkpoint: str = get_model_from_request(
        checkpoint_type="upscaler",
        fallback_model="stabilityai/stable-diffusion-x4-upscaler",
    )


class UpscalerInputData(GenerationInputData):
    init_images: list[str] = Field(
        description="Base64 encoded image to upscale"
    )
    noise_level: int = frozen_args.noise_level
    override_settings: UpscalerModelOverideSettings = None


@sdapi.post(
    "/v1/upscaler",
    summary="Does image upscaling",
    response_model=GenerationResponseData,
)
def upscaler_api(
    InputData: UpscalerInputData,
):
    model_id = get_model_from_request(
        InputData,
        checkpoint_type="upscaler",
        fallback_model="stabilityai/stable-diffusion-x4-upscaler",
    )
    scheduler = get_scheduler_from_request(InputData, "upscaler")
    (lora_weights, lora_hf_id) = get_lora_params(frozen_args.use_lora)

    init_image = decode_base64_to_image(InputData.init_images[0])

    print(
        f"Prompt: {InputData.prompt}, "
        f"Negative Prompt: {InputData.negative_prompt}, "
        f"Seed: {InputData.seed}, "
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}."
    )

    res = upscaler_inf(
        InputData.prompt,
        InputData.negative_prompt,
        init_image,
        InputData.height,
        InputData.width,
        InputData.steps,
        InputData.noise_level,
        InputData.cfg_scale,
        InputData.seed,
        batch_count=InputData.n_iter,
        batch_size=1,
        scheduler=scheduler,
        model_id=model_id,
        custom_vae=frozen_args.custom_vae or "None",
        precision="fp16",
        device=get_device(frozen_args.device),
        max_length=frozen_args.max_length,
        save_metadata_to_json=frozen_args.save_metadata_to_json,
        save_metadata_to_png=frozen_args.write_metadata_to_png,
        lora_weights=lora_weights,
        lora_hf_id=lora_hf_id,
        ondemand=frozen_args.ondemand,
        repeatable_seeds=False,
    )

    # Since we're not streaming we just want the last generator result
    for items_so_far in res:
        items = items_so_far

    return {
        "images": encode_pil_to_base64(items[0]),
        "parameters": {},
        "info": items[1],
    }
