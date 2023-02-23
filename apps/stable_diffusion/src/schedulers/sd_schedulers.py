from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DEISMultistepScheduler,
)
from apps.stable_diffusion.src.schedulers.shark_eulerdiscrete import (
    SharkEulerDiscreteScheduler,
)


def get_schedulers(model_id):
    schedulers = dict()
    schedulers["PNDM"] = PNDMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["KDPM2Discrete"] = KDPM2DiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["LMSDiscrete"] = LMSDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["DDIM"] = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers[
        "DPMSolverMultistep"
    ] = DPMSolverMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["EulerDiscrete"] = EulerDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers[
        "EulerAncestralDiscrete"
    ] = EulerAncestralDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers[
        "DEISMultistep"
    ] = DEISMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers[
        "SharkEulerDiscrete"
    ] = SharkEulerDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["SharkEulerDiscrete"].compile()
    return schedulers
