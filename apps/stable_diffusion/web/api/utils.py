import base64
import pickle

from argparse import Namespace
from fastapi.exceptions import HTTPException
from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field

from apps.stable_diffusion.src import args
from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    get_custom_model_files,
    predefined_models,
    predefined_paint_models,
    predefined_upscaler_models,
    scheduler_list,
    scheduler_list_cpu_only,
)


# Probably overly cautious, but try to ensure we only use the starting
# args in each api call, as the code does `args.<whatever> = <changed_value>`
# in lots of places and in testing, it seemed to me, these changes leaked
# into subsequent api calls.

# Roundtripping through pickle for deepcopy, there is probably a better way
frozen_args = Namespace(**(pickle.loads(pickle.dumps(vars(args)))))

# an attempt to map some of the A1111 sampler names to scheduler names
# https://github.com/huggingface/diffusers/issues/4167 is where the
# (not so obvious) ones come from
sampler_aliases = {
    # a1111/onnx (these point to diffusers classes in A1111)
    "pndm": "PNDM",
    "heun": "HeunDiscrete",
    "ddim": "DDIM",
    "ddpm": "DDPM",
    "euler": "EulerDiscrete",
    "euler-ancestral": "EulerAncestralDiscrete",
    "dpm": "DPMSolverMultistep",
    # a1111/k_diffusion (the obvious ones)
    "Euler a": "EulerAncestralDiscrete",
    "Euler": "EulerDiscrete",
    "LMS": "LMSDiscrete",
    "Heun": "HeunDiscrete",
    # a1111/k_diffusion (not so obvious)
    "DPM++ 2M": "DPMSolverMultistep",
    "DPM++ 2M Karras": "DPMSolverMultistepKarras",
    "DPM++ 2M SDE": "DPMSolverMultistep++",
    "DPM++ 2M SDE Karras": "DPMSolverMultistepKarras++",
    "DPM2": "KDPM2Discrete",
    "DPM2 a": "KDPM2AncestralDiscrete",
}

allowed_schedulers = {
    "txt2img": {
        "schedulers": scheduler_list,
        "fallback": "SharkEulerDiscrete",
    },
    "txt2img_hires": {
        "schedulers": scheduler_list_cpu_only,
        "fallback": "DEISMultistep",
    },
    "img2img": {
        "schedulers": scheduler_list_cpu_only,
        "fallback": "EulerDiscrete",
    },
    "inpaint": {
        "schedulers": scheduler_list_cpu_only,
        "fallback": "DDIM",
    },
    "outpaint": {
        "schedulers": scheduler_list_cpu_only,
        "fallback": "DDIM",
    },
    "upscaler": {
        "schedulers": scheduler_list_cpu_only,
        "fallback": "DDIM",
    },
}

# base pydantic model for sd generation apis


class GenerationInputData(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    hf_model_id: str | None = None
    height: int = Field(
        default=frozen_args.height, ge=128, le=768, multiple_of=8
    )
    width: int = Field(
        default=frozen_args.width, ge=128, le=768, multiple_of=8
    )
    sampler_name: str = frozen_args.scheduler
    cfg_scale: float = Field(default=frozen_args.guidance_scale, ge=1)
    steps: int = Field(default=frozen_args.steps, ge=1, le=100)
    seed: int = frozen_args.seed
    n_iter: int = Field(default=frozen_args.batch_count)


class GenerationResponseData(BaseModel):
    images: list[str] = Field(description="Generated images, Base64 encoded")
    properties: dict = {}
    info: str


# image encoding/decoding


def encode_pil_to_base64(images: list[Image.Image]):
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


def decode_base64_to_image(encoding: str):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";", 1)[1].split(",", 1)[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        print(err)
        raise HTTPException(status_code=400, detail="Invalid encoded image")


# get valid sd models/vaes/schedulers etc.


def get_predefined_models(custom_checkpoint_type: str):
    match custom_checkpoint_type:
        case "inpainting":
            return predefined_paint_models
        case "upscaler":
            return predefined_upscaler_models
        case _:
            return predefined_models


def get_model_from_request(
    request_data=None,
    checkpoint_type: str = "",
    fallback_model: str = "",
):
    model = None
    if request_data:
        if request_data.hf_model_id:
            model = request_data.hf_model_id
        elif request_data.override_settings:
            model = request_data.override_settings.sd_model_checkpoint

    # if the request didn't specify a model try the command line args
    result = model or frozen_args.ckpt_loc or frozen_args.hf_model_id

    # make sure whatever we have is a valid model for the checkpoint type
    if result in get_custom_model_files(
        custom_checkpoint_type=checkpoint_type
    ) + get_predefined_models(checkpoint_type):
        return result
    # if not return what was specified as the fallback
    else:
        return fallback_model


def get_scheduler_from_request(
    request_data: GenerationInputData, operation: str
):
    allowed = allowed_schedulers[operation]

    requested = request_data.sampler_name
    requested = sampler_aliases.get(requested, requested)

    return (
        requested
        if requested in allowed["schedulers"]
        else allowed["fallback"]
    )


def get_device(device_str: str):
    # first substring match in the list available devices, with first
    # device when none are matched
    return next(
        (device for device in available_devices if device_str in device),
        available_devices[0],
    )
