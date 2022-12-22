from transformers import CLIPTokenizer
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from models.stable_diffusion.opt_params import get_unet, get_vae, get_clip
from models.stable_diffusion.utils import set_init_device_flags
from models.stable_diffusion.stable_args import args
from models.stable_diffusion.schedulers import (
    SharkEulerDiscreteScheduler,
)

# initial settings.
set_init_device_flags()

model_config = {
    "v2_1": "stabilityai/stable-diffusion-2-1",
    "v2_1base": "stabilityai/stable-diffusion-2-1-base",
    "v1_4": "CompVis/stable-diffusion-v1-4",
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
if args.version != "v1_4":
    cache_obj["tokenizer"] = CLIPTokenizer.from_pretrained(
        model_config[args.version], subfolder="tokenizer"
    )
