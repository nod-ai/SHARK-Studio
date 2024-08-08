import base64

from fastapi import FastAPI

from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field
from fastapi.exceptions import HTTPException

from apps.shark_studio.api.sd import shark_sd_fn

sdapi = FastAPI()


class GenerationInputData(BaseModel):
    prompt: list = [""]
    negative_prompt: list = [""]
    hf_model_id: str | None = None
    height: int = Field(default=512, ge=128, le=1024, multiple_of=8)
    width: int = Field(default=512, ge=128, le=1024, multiple_of=8)
    sampler_name: str = "EulerDiscrete"
    cfg_scale: float = Field(default=7.5, ge=1)
    steps: int = Field(default=20, ge=1, le=100)
    seed: int = Field(default=-1)
    n_iter: int = Field(default=1)
    config: dict = None


class GenerationResponseData(BaseModel):
    images: list[str] = Field(description="Generated images, Base64 encoded")
    properties: dict = {}
    info: str


def encode_pil_to_base64(images: list[Image.Image]):
    encoded_imgs = []
    for image in images:
        with BytesIO() as output_bytes:
            image.save(output_bytes, format="PNG")
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


@sdapi.post(
    "/v1/txt2img",
    summary="Does text to image generation",
    response_model=GenerationResponseData,
)
def txt2img_api(InputData: GenerationInputData):
    model_id = (
        InputData.hf_model_id or "stabilityai/stable-diffusion-3-medium-diffusers"
    )
    scheduler = "FlowEulerDiscrete"
    print(
        f"Prompt: {InputData.prompt}, "
        f"Negative Prompt: {InputData.negative_prompt}, "
        f"Seed: {InputData.seed},"
        f"Model: {model_id}, "
        f"Scheduler: {scheduler}. "
    )
    if not getattr(InputData, "config"):
        InputData.config = {
            "precision": "fp16",
            "device": "rocm",
            "target_triple": "gfx1150",
        }

    res = shark_sd_fn(
        InputData.prompt,
        InputData.negative_prompt,
        None,
        InputData.height,
        InputData.width,
        InputData.steps,
        None,
        InputData.cfg_scale,
        InputData.seed,
        custom_vae=None,
        batch_count=InputData.n_iter,
        batch_size=1,
        scheduler=scheduler,
        base_model_id=model_id,
        custom_weights=None,
        precision=InputData.config["precision"],
        device=InputData.config["device"],
        target_triple=InputData.config["target_triple"],
        output_type="pil",
        ondemand=False,
        compiled_pipeline=False,
        resample_type=None,
        controlnets=[],
        embeddings=[],
    )

    # Since we're not streaming we just want the last generator result
    for items_so_far in res:
        items = items_so_far

    return {
        "images": encode_pil_to_base64(items[0]),
        "parameters": {},
        "info": items[1],
    }
