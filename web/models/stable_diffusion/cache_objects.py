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
schedulers["EulerDiscrete"] = EulerDiscreteScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="scheduler",
)

schedulers2 = dict()
schedulers2["PNDM"] = PNDMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="scheduler",
)
schedulers2["LMSDiscrete"] = LMSDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="scheduler",
)
schedulers2["DDIM"] = DDIMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="scheduler",
)
schedulers2[
    "DPMSolverMultistep"
] = DPMSolverMultistepScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="scheduler",
)
schedulers2["EulerDiscrete"] = EulerDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="scheduler",
)

# set iree-runtime flags
set_iree_runtime_flags(args)
args.version = "v1.4"

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

args.version = "v2.1base"
# cache tokenizer
cache_obj["tokenizer2"] = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer"
)

# cache vae, unet and clip.
(
    cache_obj["vae2"],
    cache_obj["unet2"],
    cache_obj["clip2"],
) = (get_vae(args), get_unet(args), get_clip(args))
