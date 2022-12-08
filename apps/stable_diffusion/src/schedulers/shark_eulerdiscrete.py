import sys
import numpy as np
from typing import List, Optional, Tuple, Union
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from diffusers.configuration_utils import register_to_config
from ..utils import compile_through_fx, get_shark_model, args
import torch


class SharkEulerDiscreteScheduler(EulerDiscreteScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
    ):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            prediction_type,
        )

    def compile(self):
        SCHEDULER_BUCKET = "gs://shark_tank/stable_diffusion/schedulers"
        BATCH_SIZE = args.batch_size

        model_input = {
            "euler": {
                "latent": torch.randn(
                    BATCH_SIZE, 4, args.height // 8, args.width // 8
                ),
                "output": torch.randn(
                    BATCH_SIZE, 4, args.height // 8, args.width // 8
                ),
                "sigma": torch.tensor(1).to(torch.float32),
                "dt": torch.tensor(1).to(torch.float32),
            },
        }

        example_latent = model_input["euler"]["latent"]
        example_output = model_input["euler"]["output"]
        if args.precision == "fp16":
            example_latent = example_latent.half()
            example_output = example_output.half()
        example_sigma = model_input["euler"]["sigma"]
        example_dt = model_input["euler"]["dt"]

        class ScalingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, latent, sigma):
                return latent / ((sigma**2 + 1) ** 0.5)

        class SchedulerStepModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, noise_pred, sigma, latent, dt):
                pred_original_sample = latent - sigma * noise_pred
                derivative = (latent - pred_original_sample) / sigma
                return latent + derivative * dt

        iree_flags = []
        if len(args.iree_vulkan_target_triple) > 0:
            iree_flags.append(
                f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
            )
        # Disable bindings fusion to work with moltenVK.
        if sys.platform == "darwin":
            iree_flags.append("-iree-stream-fuse-binding=false")

        if args.import_mlir:
            scaling_model = ScalingModel()
            self.scaling_model = compile_through_fx(
                scaling_model,
                (example_latent, example_sigma),
                model_name=f"euler_scale_model_input_{BATCH_SIZE}_{args.height}_{args.width}"
                + args.precision,
                extra_args=iree_flags,
            )

            step_model = SchedulerStepModel()
            self.step_model = compile_through_fx(
                step_model,
                (example_output, example_sigma, example_latent, example_dt),
                model_name=f"euler_step_{BATCH_SIZE}_{args.height}_{args.width}"
                + args.precision,
                extra_args=iree_flags,
            )
        else:
            self.scaling_model = get_shark_model(
                SCHEDULER_BUCKET,
                "euler_scale_model_input_" + args.precision,
                iree_flags,
            )
            self.step_model = get_shark_model(
                SCHEDULER_BUCKET, "euler_step_" + args.precision, iree_flags
            )

    def scale_model_input(self, sample, timestep):
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        return self.scaling_model(
            "forward",
            (
                sample,
                sigma,
            ),
            send_to_host=False,
        )

    def step(self, noise_pred, timestep, latent):
        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]
        dt = self.sigmas[step_index + 1] - sigma
        return self.step_model(
            "forward",
            (
                noise_pred,
                sigma,
                latent,
                dt,
            ),
            send_to_host=False,
        )
