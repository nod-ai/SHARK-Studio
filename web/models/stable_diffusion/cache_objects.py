from transformers import CLIPTokenizer
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from models.stable_diffusion.opt_params import get_unet, get_vae, get_clip
from models.stable_diffusion.utils import set_iree_runtime_flags
from models.stable_diffusion.stable_args import args
from models.stable_diffusion.schedulers import (
    SharkEulerDiscreteScheduler,
)
import os

os.environ["AMD_ENABLE_LLPC"] = "1"

# set iree-runtime flags
set_iree_runtime_flags()

model_config = {
    "v2": "stabilityai/stable-diffusion-2",
    "v2.1base": "stabilityai/stable-diffusion-2-1-base",
    "v1.4": "CompVis/stable-diffusion-v1-4",
}

schedulers = dict()
schedulers["PNDM"] = PNDMScheduler.from_pretrained(
    model_config[args.version],
    subfolder="scheduler",
)
schedulers["LMSDiscrete"] = LMSDiscreteScheduler.from_pretrained(
    model_config[args.version],
    subfolder="scheduler",
)
schedulers["DDIM"] = DDIMScheduler.from_pretrained(
    model_config[args.version],
    subfolder="scheduler",
)
schedulers["DPMSolverMultistep"] = DPMSolverMultistepScheduler.from_pretrained(
    model_config[args.version],
    subfolder="scheduler",
)
schedulers["EulerDiscrete"] = EulerDiscreteScheduler.from_pretrained(
    model_config[args.version],
    subfolder="scheduler",
)
schedulers["SharkEulerDiscrete"] = SharkEulerDiscreteScheduler.from_pretrained(
    model_config[args.version],
    subfolder="scheduler",
)
schedulers["SharkEulerDiscrete"].compile()

cache_obj = dict()
# cache vae, unet and clip.
(
    cache_obj["vae"],
    cache_obj["unet"],
    cache_obj["clip"],
) = (get_vae(), get_unet(), get_clip())

# cache tokenizer
cache_obj["tokenizer"] = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14"
)
if args.version == "v2.1base":
    cache_obj["tokenizer"] = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer"
    )
