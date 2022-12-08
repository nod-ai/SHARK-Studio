from transformers import CLIPTokenizer
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
from models.stable_diffusion.opt_params import get_unet, get_vae, get_clip
from models.stable_diffusion.utils import set_iree_runtime_flags
from models.stable_diffusion.stable_args import args


schedulers = dict()
schedulers["PNDM"] = PNDMScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="scheduler",
)
schedulers["LMSDiscrete"] = LMSDiscreteScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="scheduler",
)
schedulers["DDIM"] = DDIMScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="scheduler",
)
schedulers["DPMSolverMultistep"] = DPMSolverMultistepScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="scheduler",
)

# set iree-runtime flags
set_iree_runtime_flags(args)

cache_obj = dict()

# cache tokenizer
cache_obj["tokenizer"] = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14"
)

# cache vae, unet and clip.
(
    cache_obj["vae"],
    cache_obj["unet"],
    cache_obj["clip"],
) = (get_vae(args), get_unet(args), get_clip(args))
