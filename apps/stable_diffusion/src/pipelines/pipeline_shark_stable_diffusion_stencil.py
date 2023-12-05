import torch
import time
import numpy as np
from tqdm.auto import tqdm
from random import randint
from PIL import Image
from transformers import CLIPTokenizer
from typing import Union
from shark.shark_inference import SharkInference
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DDPMScheduler,
    KDPM2DiscreteScheduler,
)
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
    StableDiffusionPipeline,
)
from apps.stable_diffusion.src.utils import (
    controlnet_hint_conversion,
    controlnet_hint_reshaping,
)
from apps.stable_diffusion.src.utils import (
    start_profiling,
    end_profiling,
)
from apps.stable_diffusion.src.utils import (
    resamplers,
    resampler_list,
)
from apps.stable_diffusion.src.models import (
    SharkifyStableDiffusionModel,
    get_vae_encode,
)


class StencilPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            SharkEulerDiscreteScheduler,
            DEISMultistepScheduler,
            DPMSolverSinglestepScheduler,
            KDPM2AncestralDiscreteScheduler,
            HeunDiscreteScheduler,
            DDPMScheduler,
            KDPM2DiscreteScheduler,
        ],
        sd_model: SharkifyStableDiffusionModel,
        import_mlir: bool,
        use_lora: str,
        lora_strength: float,
        ondemand: bool,
        controlnet_names: list[str],
    ):
        super().__init__(
            scheduler, sd_model, import_mlir, use_lora, lora_strength, ondemand
        )
        self.controlnet = [None] * len(controlnet_names)
        self.controlnet_512 = [None] * len(controlnet_names)
        self.controlnet_id = [str] * len(controlnet_names)
        self.controlnet_512_id = [str] * len(controlnet_names)
        self.controlnet_names = controlnet_names
        self.vae_encode = None

    def load_vae_encode(self):
        if self.vae_encode is not None:
            return

        if self.import_mlir or self.use_lora:
            self.vae_encode = self.sd_model.vae_encode()
        else:
            try:
                self.vae_encode = get_vae_encode()
            except:
                print("download pipeline failed, falling back to import_mlir")
                self.vae_encode = self.sd_model.vae_encode()

    def unload_vae_encode(self):
        del self.vae_encode
        self.vae_encode = None

    def load_controlnet(self, index, model_name):
        if model_name is None:
            return
        if (
            self.controlnet[index] is not None
            and self.controlnet_id[index] is not None
            and self.controlnet_id[index] == model_name
        ):
            return
        self.controlnet_id[index] = model_name
        self.controlnet[index] = self.sd_model.controlnet(model_name)

    def unload_controlnet(self, index):
        del self.controlnet[index]
        self.controlnet_id[index] = None
        self.controlnet[index] = None

    def load_controlnet_512(self, index, model_name):
        if (
            self.controlnet_512[index] is not None
            and self.controlnet_512_id[index] == model_name
        ):
            return
        self.controlnet_512_id[index] = model_name
        self.controlnet_512[index] = self.sd_model.controlnet(
            model_name, use_large=True
        )

    def unload_controlnet_512(self, index):
        del self.controlnet_512[index]
        self.controlnet_512_id[index] = None
        self.controlnet_512[index] = None

    def prepare_latents(
        self,
        batch_size,
        height,
        width,
        generator,
        num_inference_steps,
        dtype,
    ):
        latents = torch.randn(
            (
                batch_size,
                4,
                height // 8,
                width // 8,
            ),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.is_scale_input_called = True
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(
        self,
        image,
        batch_size,
        height,
        width,
        generator,
        num_inference_steps,
        strength,
        dtype,
        resample_type,
    ):
        # Pre process image -> get image encoded -> process latents

        # TODO: process with variable HxW combos

        # Pre-process image
        resample_type = (
            resamplers[resample_type]
            if resample_type in resampler_list
            # Fallback to Lanczos
            else Image.Resampling.LANCZOS
        )

        image = image.resize((width, height), resample=resample_type)
        image_arr = np.stack([np.array(i) for i in (image,)], axis=0)
        image_arr = image_arr / 255.0
        image_arr = torch.from_numpy(image_arr).permute(0, 3, 1, 2).to(dtype)
        image_arr = 2 * (image_arr - 0.5)

        # set scheduler steps
        self.scheduler.set_timesteps(num_inference_steps)
        init_timestep = min(
            int(num_inference_steps * strength), num_inference_steps
        )
        t_start = max(num_inference_steps - init_timestep, 0)
        # timesteps reduced as per strength
        timesteps = self.scheduler.timesteps[t_start:]
        # new number of steps to be used as per strength will be
        # num_inference_steps = num_inference_steps - t_start

        # image encode
        latents = self.encode_image((image_arr,))
        latents = torch.from_numpy(latents).to(dtype)
        # add noise to data
        noise = torch.randn(latents.shape, generator=generator, dtype=dtype)
        latents = self.scheduler.add_noise(
            latents, noise, timesteps[0].repeat(1)
        )

        return latents, timesteps

    def produce_stencil_latents(
        self,
        latents,
        text_embeddings,
        guidance_scale,
        total_timesteps,
        dtype,
        cpu_scheduling,
        stencil_hints=[None],
        controlnet_conditioning_scale: float = 1.0,
        control_mode="Balanced",  # Prompt, Balanced, or Controlnet
        mask=None,
        masked_image_latents=None,
        return_all_latents=False,
    ):
        step_time_sum = 0
        latent_history = [latents]
        text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
        text_embeddings_numpy = text_embeddings.detach().numpy()
        assert control_mode in ["Prompt", "Balanced", "Controlnet"]
        if text_embeddings.shape[1] <= self.model_max_length:
            self.load_unet()
        else:
            self.load_unet_512()

        for i, name in enumerate(self.controlnet_names):
            use_names = []
            if name is not None:
                use_names.append(name)
            else:
                continue
            if text_embeddings.shape[1] <= self.model_max_length:
                self.load_controlnet(i, name)
            else:
                self.load_controlnet_512(i, name)
            self.controlnet_names = use_names

        for i, t in tqdm(enumerate(total_timesteps)):
            step_start_time = time.time()
            timestep = torch.tensor([t]).to(dtype)
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            if mask is not None and masked_image_latents is not None:
                latent_model_input = torch.cat(
                    [
                        torch.from_numpy(np.asarray(latent_model_input)),
                        mask,
                        masked_image_latents,
                    ],
                    dim=1,
                ).to(dtype)
            if cpu_scheduling:
                latent_model_input = latent_model_input.detach().numpy()

            if not torch.is_tensor(latent_model_input):
                latent_model_input_1 = torch.from_numpy(
                    np.asarray(latent_model_input)
                ).to(dtype)
            else:
                latent_model_input_1 = latent_model_input

            # Multicontrolnet
            width = latent_model_input_1.shape[2]
            height = latent_model_input_1.shape[3]
            dtype = latent_model_input_1.dtype
            control_acc = (
                [torch.zeros((2, 320, height, width), dtype=dtype)] * 3
                + [
                    torch.zeros(
                        (2, 320, int(height / 2), int(width / 2)), dtype=dtype
                    )
                ]
                + [
                    torch.zeros(
                        (2, 640, int(height / 2), int(width / 2)), dtype=dtype
                    )
                ]
                * 2
                + [
                    torch.zeros(
                        (2, 640, int(height / 4), int(width / 4)), dtype=dtype
                    )
                ]
                + [
                    torch.zeros(
                        (2, 1280, int(height / 4), int(width / 4)), dtype=dtype
                    )
                ]
                * 2
                + [
                    torch.zeros(
                        (2, 1280, int(height / 8), int(width / 8)), dtype=dtype
                    )
                ]
                * 4
            )
            for i, controlnet_hint in enumerate(stencil_hints):
                if controlnet_hint is None:
                    pass
                if text_embeddings.shape[1] <= self.model_max_length:
                    control = self.controlnet[i](
                        "forward",
                        (
                            latent_model_input_1,
                            timestep,
                            text_embeddings,
                            controlnet_hint,
                            *control_acc,
                        ),
                        send_to_host=False,
                    )
                else:
                    control = self.controlnet_512[i](
                        "forward",
                        (
                            latent_model_input_1,
                            timestep,
                            text_embeddings,
                            controlnet_hint,
                            *control_acc,
                        ),
                        send_to_host=False,
                    )
                control_acc = control[13:]
                control = control[:13]

            timestep = timestep.detach().numpy()
            # Profiling Unet.
            profile_device = start_profiling(file_path="unet.rdc")
            # TODO: Pass `control` as it is to Unet. Same as TODO mentioned in model_wrappers.py.

            dtype = latents.dtype
            if control_mode == "Balanced":
                control_scale = [
                    torch.tensor(1.0, dtype=dtype) for _ in range(len(control))
                ]
            elif control_mode == "Prompt":
                control_scale = [
                    torch.tensor(0.825**x, dtype=dtype)
                    for x in range(len(control))
                ]
            elif control_mode == "Controlnet":
                control_scale = [
                    torch.tensor(float(guidance_scale), dtype=dtype)
                    for _ in range(len(control))
                ]

            if text_embeddings.shape[1] <= self.model_max_length:
                noise_pred = self.unet(
                    "forward",
                    (
                        latent_model_input,
                        timestep,
                        text_embeddings_numpy,
                        guidance_scale,
                        control[0],
                        control[1],
                        control[2],
                        control[3],
                        control[4],
                        control[5],
                        control[6],
                        control[7],
                        control[8],
                        control[9],
                        control[10],
                        control[11],
                        control[12],
                        control_scale[0],
                        control_scale[1],
                        control_scale[2],
                        control_scale[3],
                        control_scale[4],
                        control_scale[5],
                        control_scale[6],
                        control_scale[7],
                        control_scale[8],
                        control_scale[9],
                        control_scale[10],
                        control_scale[11],
                        control_scale[12],
                    ),
                    send_to_host=False,
                )
            else:
                noise_pred = self.unet_512(
                    "forward",
                    (
                        latent_model_input,
                        timestep,
                        text_embeddings_numpy,
                        guidance_scale,
                        control[0],
                        control[1],
                        control[2],
                        control[3],
                        control[4],
                        control[5],
                        control[6],
                        control[7],
                        control[8],
                        control[9],
                        control[10],
                        control[11],
                        control[12],
                        control_scale[0],
                        control_scale[1],
                        control_scale[2],
                        control_scale[3],
                        control_scale[4],
                        control_scale[5],
                        control_scale[6],
                        control_scale[7],
                        control_scale[8],
                        control_scale[9],
                        control_scale[10],
                        control_scale[11],
                        control_scale[12],
                    ),
                    send_to_host=False,
                )
            end_profiling(profile_device)

            if cpu_scheduling:
                noise_pred = torch.from_numpy(noise_pred.to_host())
                latents = self.scheduler.step(
                    noise_pred, t, latents
                ).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents)

            latent_history.append(latents)
            step_time = (time.time() - step_start_time) * 1000
            #  self.log += (
            #      f"\nstep = {i} | timestep = {t} | time = {step_time:.2f}ms"
            #  )
            step_time_sum += step_time

        if self.ondemand:
            self.unload_unet()
            self.unload_unet_512()
            for i in range(len(self.controlnet_names)):
                self.unload_controlnet(i)
                self.unload_controlnet_512(i)
        avg_step_time = step_time_sum / len(total_timesteps)
        self.log += f"\nAverage step time: {avg_step_time}ms/it"

        if not return_all_latents:
            return latents
        all_latents = torch.cat(latent_history, dim=0)
        return all_latents

    def encode_image(self, input_image):
        self.load_vae_encode()
        vae_encode_start = time.time()
        latents = self.vae_encode("forward", input_image)
        vae_inf_time = (time.time() - vae_encode_start) * 1000
        if self.ondemand:
            self.unload_vae_encode()
        self.log += f"\nVAE Encode Inference time (ms): {vae_inf_time:.3f}"

        return latents

    def generate_images(
        self,
        prompts,
        neg_prompts,
        image,
        batch_size,
        height,
        width,
        num_inference_steps,
        strength,
        guidance_scale,
        seed,
        max_length,
        dtype,
        use_base_vae,
        cpu_scheduling,
        max_embeddings_multiples,
        stencils,
        stencil_images,
        resample_type,
        control_mode,
        preprocessed_hints,
    ):
        # Control Embedding check & conversion
        # controlnet_hint = controlnet_hint_conversion(
        #     image, use_stencil, height, width, dtype, num_images_per_prompt=1
        # )
        stencil_hints = []
        self.sd_model.stencils = stencils
        for i, hint in enumerate(preprocessed_hints):
            if hint is not None:
                hint = controlnet_hint_reshaping(
                    hint,
                    height,
                    width,
                    dtype,
                    num_images_per_prompt=1,
                )
                stencil_hints.append(hint)

        for i, stencil in enumerate(stencils):
            if stencil == None:
                continue
            if len(stencil_hints) > i:
                if stencil_hints[i] is not None:
                    print(f"Using preprocessed controlnet hint for {stencil}")
                    continue
            image = stencil_images[i]
            stencil_hints.append(
                controlnet_hint_conversion(
                    image,
                    stencil,
                    height,
                    width,
                    dtype,
                    num_images_per_prompt=1,
                )
            )

        # prompts and negative prompts must be a list.
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(neg_prompts, str):
            neg_prompts = [neg_prompts]

        prompts = prompts * batch_size
        neg_prompts = neg_prompts * batch_size

        # seed generator to create the inital latent noise. Also handle out of range seeds.
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        if seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)
        generator = torch.manual_seed(seed)

        # Get text embeddings with weight emphasis from prompts
        text_embeddings = self.encode_prompts_weight(
            prompts,
            neg_prompts,
            max_length,
            max_embeddings_multiples=max_embeddings_multiples,
        )

        # guidance scale as a float32 tensor.
        guidance_scale = torch.tensor(guidance_scale).to(torch.float32)
        if image is not None:
            # Prepare input image latent
            init_latents, final_timesteps = self.prepare_image_latents(
                image=image,
                batch_size=batch_size,
                height=height,
                width=width,
                generator=generator,
                num_inference_steps=num_inference_steps,
                strength=strength,
                dtype=dtype,
                resample_type=resample_type,
            )
        else:
            # Prepare initial latent.
            init_latents = self.prepare_latents(
                batch_size=batch_size,
                height=height,
                width=width,
                generator=generator,
                num_inference_steps=num_inference_steps,
                dtype=dtype,
            )
            final_timesteps = self.scheduler.timesteps

        # Get Image latents
        latents = self.produce_stencil_latents(
            latents=init_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            total_timesteps=final_timesteps,
            dtype=dtype,
            cpu_scheduling=cpu_scheduling,
            control_mode=control_mode,
            stencil_hints=stencil_hints,
        )

        # Img latents -> PIL images
        all_imgs = []
        self.load_vae()
        for i in tqdm(range(0, latents.shape[0], batch_size)):
            imgs = self.decode_latents(
                latents=latents[i : i + batch_size],
                use_base_vae=use_base_vae,
                cpu_scheduling=cpu_scheduling,
            )
            all_imgs.extend(imgs)
        if self.ondemand:
            self.unload_vae()

        return all_imgs
