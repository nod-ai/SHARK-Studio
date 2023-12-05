from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
)
from apps.stable_diffusion.src.schedulers.shark_eulerdiscrete import (
    SharkEulerDiscreteScheduler,
)
from apps.stable_diffusion.src.schedulers.shark_eulerancestraldiscrete import (
    SharkEulerAncestralDiscreteScheduler,
)


def get_schedulers(model_id):
    # TODO: Robust scheduler setup on pipeline creation -- if we don't
    # set batch_size here, the SHARK schedulers will
    # compile with batch size = 1 regardless of whether the model
    # outputs latents of a larger batch size, e.g. SDXL.
    # This also goes towards enabling batch size cfg for SD in general.
    # However, obviously, searching for whether the base model ID
    # contains "xl" is not very robust.

    batch_size = 2 if "xl" in model_id.lower() else 1

    schedulers = dict()
    schedulers["PNDM"] = PNDMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["DDPM"] = DDPMScheduler.from_pretrained(
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
        model_id, subfolder="scheduler", algorithm_type="dpmsolver"
    )
    schedulers[
        "DPMSolverMultistep++"
    ] = DPMSolverMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler", algorithm_type="dpmsolver++"
    )
    schedulers[
        "DPMSolverMultistepKarras"
    ] = DPMSolverMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        use_karras_sigmas=True,
    )
    schedulers[
        "DPMSolverMultistepKarras++"
    ] = DPMSolverMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
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
    schedulers["DEISMultistep"] = DEISMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers[
        "SharkEulerDiscrete"
    ] = SharkEulerDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers[
        "SharkEulerAncestralDiscrete"
    ] = SharkEulerAncestralDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers[
        "DPMSolverSinglestep"
    ] = DPMSolverSinglestepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers[
        "KDPM2AncestralDiscrete"
    ] = KDPM2AncestralDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["HeunDiscrete"] = HeunDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["SharkEulerDiscrete"].compile(batch_size)
    schedulers["SharkEulerAncestralDiscrete"].compile(batch_size)
    return schedulers
