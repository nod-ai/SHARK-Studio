from transformers import CLIPTokenizer
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
from models.stable_diffusion.opt_params import get_unet, get_vae, get_clip
from models.stable_diffusion.utils import (
    set_init_device_flags,
    set_iree_runtime_flags,
)
from models.stable_diffusion.stable_args import args
from models.stable_diffusion.schedulers import (
    SharkEulerDiscreteScheduler,
)


# set iree runtime flags. This should be the very first thing in the program.
set_iree_runtime_flags()

model_config = {
    "v2_1": "stabilityai/stable-diffusion-2-1",
    "v2_1base": "stabilityai/stable-diffusion-2-1-base",
    "v1_4": "CompVis/stable-diffusion-v1-4",
}


def get_schedulers(version):
    schedulers = dict()
    schedulers["PNDM"] = PNDMScheduler.from_pretrained(
        model_config[version],
        subfolder="scheduler",
    )
    schedulers["LMSDiscrete"] = LMSDiscreteScheduler.from_pretrained(
        model_config[version],
        subfolder="scheduler",
    )
    schedulers["DDIM"] = DDIMScheduler.from_pretrained(
        model_config[version],
        subfolder="scheduler",
    )
    schedulers[
        "DPMSolverMultistep"
    ] = DPMSolverMultistepScheduler.from_pretrained(
        model_config[version],
        subfolder="scheduler",
    )
    schedulers["EulerDiscrete"] = EulerDiscreteScheduler.from_pretrained(
        model_config[version],
        subfolder="scheduler",
    )
    schedulers[
        "EulerAncestral"
    ] = EulerAncestralDiscreteScheduler.from_pretrained(
        model_config[version],
        subfolder="scheduler",
    )
    schedulers[
        "SharkEulerDiscrete"
    ] = SharkEulerDiscreteScheduler.from_pretrained(
        model_config[version],
        subfolder="scheduler",
    )
    schedulers["SharkEulerDiscrete"].compile()
    return schedulers


def get_tokenizer(version):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    if version != "v1_4":
        tokenizer = CLIPTokenizer.from_pretrained(
            model_config[version], subfolder="tokenizer"
        )
    return tokenizer


class ModelCache:
    def __init__(self):
        self.device = None
        self.variant = None
        self.version = None
        self.schedulers = get_schedulers(args.version)
        self.tokenizer = get_tokenizer(args.version)

    def set_models(self, device_key):
        set_iree_runtime_flags()
        if self.device != device_key or self.variant != args.variant:
            self.device = device_key
            self.variant = args.variant
            self.version = args.version
            args.device = device_key.split("=>", 1)[0].strip()
            args.use_tuned = True
            set_init_device_flags()
            self.vae = get_vae()
            self.unet = get_unet()
            self.clip = get_clip()


model_cache = ModelCache()
