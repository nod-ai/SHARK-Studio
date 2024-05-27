# from shark_turbine.turbine_models.schedulers import export_scheduler_model
from diffusers import (
    LCMScheduler,
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


def get_schedulers(model_id):
    # TODO: switch over to turbine and run all on GPU
    print(f"\n[LOG] Initializing schedulers from model id: {model_id}")
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
    schedulers["LCMScheduler"] = LCMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    schedulers["DPMSolverMultistep"] = DPMSolverMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler", algorithm_type="dpmsolver"
    )
    schedulers["DPMSolverMultistep++"] = DPMSolverMultistepScheduler.from_pretrained(
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
    schedulers["DPMSolverSinglestep"] = DPMSolverSinglestepScheduler.from_pretrained(
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
    return schedulers


def export_scheduler_model(model):
    return "None", "None"


scheduler_model_map = {
    "PNDM": export_scheduler_model("PNDMScheduler"),
    "DPMSolverSDE": export_scheduler_model("DpmSolverSDEScheduler"),
    "EulerDiscrete": export_scheduler_model("EulerDiscreteScheduler"),
    "EulerAncestralDiscrete": export_scheduler_model("EulerAncestralDiscreteScheduler"),
    "LCM": export_scheduler_model("LCMScheduler"),
    "LMSDiscrete": export_scheduler_model("LMSDiscreteScheduler"),
    "DDPM": export_scheduler_model("DDPMScheduler"),
    "DDIM": export_scheduler_model("DDIMScheduler"),
    "DPMSolverMultistep": export_scheduler_model("DPMSolverMultistepScheduler"),
    "KDPM2Discrete": export_scheduler_model("KDPM2DiscreteScheduler"),
    "DEISMultistep": export_scheduler_model("DEISMultistepScheduler"),
    "DPMSolverSinglestep": export_scheduler_model("DPMSolverSingleStepScheduler"),
    "KDPM2AncestralDiscrete": export_scheduler_model("KDPM2AncestralDiscreteScheduler"),
    "HeunDiscrete": export_scheduler_model("HeunDiscreteScheduler"),
}
