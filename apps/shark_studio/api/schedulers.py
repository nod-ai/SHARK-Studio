#from shark_turbine.turbine_models.schedulers import export_scheduler_model

def export_scheduler_model(model):
    return "None", "None"

scheduler_model_map = {
    "EulerDiscrete": export_scheduler_model("EulerDiscreteScheduler"),
    "EulerAncestralDiscrete": export_scheduler_model("EulerAncestralDiscreteScheduler"),
    "LCM": export_scheduler_model("LCMScheduler"),
    "LMSDiscrete": export_scheduler_model("LMSDiscreteScheduler"),
    "PNDM": export_scheduler_model("PNDMScheduler"),
    "DDPM": export_scheduler_model("DDPMScheduler"),
    "DDIM": export_scheduler_model("DDIMScheduler"),
    "DPMSolverMultistep": export_scheduler_model("DPMSolverMultistepScheduler"),
    "KDPM2Discrete": export_scheduler_model("KDPM2DiscreteScheduler"),
    "DEISMultistep": export_scheduler_model("DEISMultistepScheduler"),
    "DPMSolverSinglestep": export_scheduler_model("DPMSolverSingleStepScheduler"),
    "KDPM2AncestralDiscrete": export_scheduler_model("KDPM2AncestralDiscreteScheduler"),
    "HeunDiscrete": export_scheduler_model("HeunDiscreteScheduler"),
}
