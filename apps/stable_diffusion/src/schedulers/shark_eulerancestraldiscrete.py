import sys
import numpy as np
from typing import List, Optional, Tuple, Union
from diffusers import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import register_to_config
from apps.stable_diffusion.src.utils import (
    compile_through_fx,
    get_shark_model,
    args,
)
import torch


class SharkEulerAncestralDiscreteScheduler(EulerAncestralDiscreteScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            prediction_type,
            timestep_spacing,
            steps_offset,
        )
        # TODO: make it dynamic so we dont have to worry about batch size
        self.batch_size = None
        self.init_input_shape = None

    def compile(self, batch_size=1):
        SCHEDULER_BUCKET = "gs://shark_tank/stable_diffusion/schedulers"
        device = args.device.split(":", 1)[0].strip()
        self.batch_size = batch_size

        model_input = {
            "eulera": {
                "output": torch.randn(
                    batch_size, 4, args.height // 8, args.width // 8
                ),
                "latent": torch.randn(
                    batch_size, 4, args.height // 8, args.width // 8
                ),
                "sigma": torch.tensor(1).to(torch.float32),
                "sigma_from": torch.tensor(1).to(torch.float32),
                "sigma_to": torch.tensor(1).to(torch.float32),
                "noise": torch.randn(
                    batch_size, 4, args.height // 8, args.width // 8
                ),
            },
        }

        example_latent = model_input["eulera"]["latent"]
        example_output = model_input["eulera"]["output"]
        example_noise = model_input["eulera"]["noise"]
        if args.precision == "fp16":
            example_latent = example_latent.half()
            example_output = example_output.half()
            example_noise = example_noise.half()
        example_sigma = model_input["eulera"]["sigma"]
        example_sigma_from = model_input["eulera"]["sigma_from"]
        example_sigma_to = model_input["eulera"]["sigma_to"]

        class ScalingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, latent, sigma):
                return latent / ((sigma**2 + 1) ** 0.5)

        class SchedulerStepEpsilonModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self, noise_pred, latent, sigma, sigma_from, sigma_to, noise
            ):
                sigma_up = (
                    sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
                ) ** 0.5
                sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
                dt = sigma_down - sigma
                pred_original_sample = latent - sigma * noise_pred
                derivative = (latent - pred_original_sample) / sigma
                prev_sample = latent + derivative * dt
                return prev_sample + noise * sigma_up

        class SchedulerStepVPredictionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self, noise_pred, sigma, sigma_from, sigma_to, latent, noise
            ):
                sigma_up = (
                    sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
                ) ** 0.5
                sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
                dt = sigma_down - sigma
                pred_original_sample = noise_pred * (
                    -sigma / (sigma**2 + 1) ** 0.5
                ) + (latent / (sigma**2 + 1))
                derivative = (latent - pred_original_sample) / sigma
                prev_sample = latent + derivative * dt
                return prev_sample + noise * sigma_up

        iree_flags = []
        if len(args.iree_vulkan_target_triple) > 0:
            iree_flags.append(
                f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
            )

        def _import(self):
            scaling_model = ScalingModel()
            self.scaling_model, _ = compile_through_fx(
                model=scaling_model,
                inputs=(example_latent, example_sigma),
                extended_model_name=f"euler_a_scale_model_input_{self.batch_size}_{args.height}_{args.width}_{device}_"
                + args.precision,
                extra_args=iree_flags,
            )

            pred_type_model_dict = {
                "epsilon": SchedulerStepEpsilonModel(),
                "v_prediction": SchedulerStepVPredictionModel(),
            }
            step_model = pred_type_model_dict[self.config.prediction_type]
            self.step_model, _ = compile_through_fx(
                step_model,
                (
                    example_output,
                    example_latent,
                    example_sigma,
                    example_sigma_from,
                    example_sigma_to,
                    example_noise,
                ),
                extended_model_name=f"euler_a_step_{self.config.prediction_type}_{self.batch_size}_{args.height}_{args.width}_{device}_"
                + args.precision,
                extra_args=iree_flags,
            )

        if args.import_mlir:
            _import(self)

        else:
            try:
                self.scaling_model = get_shark_model(
                    SCHEDULER_BUCKET,
                    "euler_a_scale_model_input_" + args.precision,
                    iree_flags,
                )
                self.step_model = get_shark_model(
                    SCHEDULER_BUCKET,
                    "euler_a_step_"
                    + self.config.prediction_type
                    + args.precision,
                    iree_flags,
                )
            except:
                print(
                    "failed to download model, falling back and using import_mlir"
                )
                args.import_mlir = True
                _import(self)

    def scale_model_input(self, sample, timestep):
        if self.step_index is None:
            self._init_step_index(timestep)
        sigma = self.sigmas[self.step_index]
        return self.scaling_model(
            "forward",
            (
                sample,
                sigma,
            ),
            send_to_host=False,
        )

    def step(
        self,
        noise_pred,
        timestep,
        latent,
        generator: Optional[torch.Generator] = None,
        return_dict: Optional[bool] = False,
    ):
        step_inputs = []

        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        noise = randn_tensor(
            torch.Size(noise_pred.shape),
            dtype=torch.float16,
            device="cpu",
            generator=generator,
        )
        step_inputs = [
            noise_pred,
            latent,
            sigma,
            sigma_from,
            sigma_to,
            noise,
        ]
        # TODO: deal with dynamic inputs in turbine flow.
        # update step index since we're done with the variable and will return with compiled module output.
        self._step_index += 1

        if noise_pred.shape[0] < self.batch_size:
            for i in [0, 1, 5]:
                try:
                    step_inputs[i] = torch.tensor(step_inputs[i])
                except:
                    step_inputs[i] = torch.tensor(step_inputs[i].to_host())
                step_inputs[i] = torch.cat(
                    (step_inputs[i], step_inputs[i]), axis=0
                )
            return self.step_model(
                "forward",
                tuple(step_inputs),
                send_to_host=True,
            )

        return self.step_model(
            "forward",
            tuple(step_inputs),
            send_to_host=False,
        )
