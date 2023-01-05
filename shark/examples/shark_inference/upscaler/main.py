import requests
from PIL import Image
from io import BytesIO
from pipeline_shark_stable_diffusion_upscale import (
    SharkStableDiffusionUpscalePipeline,
)
import torch

model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = SharkStableDiffusionUpscalePipeline(model_id)

# let's download an  image
url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
response = requests.get(url)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = low_res_img.resize((128, 128))

prompt = "a white cat"

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("upsampled_cat.png")
