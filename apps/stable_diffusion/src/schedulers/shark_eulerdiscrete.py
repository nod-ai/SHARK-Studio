import sys
import numpy as np
from typing import List, Optional, Tuple, Union
from diffusers import (
    EulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import register_to_config
from apps.stable_diffusion.src.utils import (
    compile_through_fx,
    get_shark_model,
    args,
)
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
        interpolation_type: str = "linear",
        use_karras_sigmas: bool = False,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        timestep_spacing: str = "linspace",
        timestep_type: str = "discrete",
        steps_offset: int = 0,
    ):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            prediction_type,
            interpolation_type,
            use_karras_sigmas,
            sigma_min,
            sigma_max,
            timestep_spacing,
            timestep_type,
            steps_offset,
        )
        # TODO: make it dynamic so we dont have to worry about batch size
        self.batch_size = 1

    def compile(self, batch_size=1):
        SCHEDULER_BUCKET = "gs://shark_tank/stable_diffusion/schedulers"
        device = args.device.split(":", 1)[0].strip()
        self.batch_size = batch_size

        model_input = {
            "euler": {
                "latent": torch.randn(
                    batch_size, 4, args.height // 8, args.width // 8
                ),
                "output": torch.randn(
                    batch_size, 4, args.height // 8, args.width // 8
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

        class SchedulerStepEpsilonModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, noise_pred, sigma_hat, latent, dt):
                pred_original_sample = latent - sigma_hat * noise_pred
                derivative = (latent - pred_original_sample) / sigma_hat
                return latent + derivative * dt

        class SchedulerStepSampleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, noise_pred, sigma_hat, latent, dt):
                pred_original_sample = noise_pred
                derivative = (latent - pred_original_sample) / sigma_hat
                return latent + derivative * dt

        class SchedulerStepVPredictionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, noise_pred, sigma, latent, dt):
                pred_original_sample = noise_pred * (
                    -sigma / (sigma**2 + 1) ** 0.5
                ) + (latent / (sigma**2 + 1))
                derivative = (latent - pred_original_sample) / sigma
                return latent + derivative * dt

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
                extended_model_name=f"euler_scale_model_input_{self.batch_size}_{args.height}_{args.width}_{device}_"
                + args.precision,
                extra_args=iree_flags,
            )

            pred_type_model_dict = {
                "epsilon": SchedulerStepEpsilonModel(),
                "v_prediction": SchedulerStepVPredictionModel(),
                "sample": SchedulerStepSampleModel(),
                "original_sample": SchedulerStepSampleModel(),
            }
            step_model = pred_type_model_dict[self.config.prediction_type]
            self.step_model, _ = compile_through_fx(
                step_model,
                (example_output, example_sigma, example_latent, example_dt),
                extended_model_name=f"euler_step_{self.config.prediction_type}_{self.batch_size}_{args.height}_{args.width}_{device}_"
                + args.precision,
                extra_args=iree_flags,
            )

        if args.import_mlir:
            _import(self)

        else:
            try:
                step_model_type = (
                    "sample"
                    if "sample" in self.config.prediction_type
                    else self.config.prediction_type
                )
                self.scaling_model = get_shark_model(
                    SCHEDULER_BUCKET,
                    "euler_scale_model_input_" + args.precision,
                    iree_flags,
                )
                self.step_model = get_shark_model(
                    SCHEDULER_BUCKET,
                    "euler_step_" + step_model_type + args.precision,
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
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: Optional[bool] = False,
    ):
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]

        gamma = (
            min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigma <= s_tmax
            else 0.0
        )

        sigma_hat = sigma * (gamma + 1)

        noise_pred = (
            torch.from_numpy(noise_pred)
            if isinstance(noise_pred, np.ndarray)
            else noise_pred
        )

        noise = randn_tensor(
            torch.Size(noise_pred.shape),
            dtype=torch.float16,
            device="cpu",
            generator=generator,
        )

        eps = noise * s_noise

        if gamma > 0:
            latent = latent + eps * (sigma_hat**2 - sigma**2) ** 0.5

        if self.config.prediction_type == "v_prediction":
            sigma_hat = sigma

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        self._step_index += 1

        return self.step_model(
            "forward",
            (
                noise_pred,
                sigma_hat,
                latent,
                dt,
            ),
            send_to_host=False,
        )
