import inspect
import torch
import time
from tqdm.auto import tqdm
import numpy as np
from random import randint
from transformers import CLIPTokenizer
from typing import Union
from shark.shark_inference import SharkInference
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
)
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
    StableDiffusionPipeline,
)
from apps.stable_diffusion.src.utils import (
    start_profiling,
    end_profiling,
)
from PIL import Image
from apps.stable_diffusion.src.models import SharkifyStableDiffusionModel


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        w, h = image[0].size
        w, h = map(
            lambda x: x - x % 64, (w, h)
        )  # resize to integer multiple of 64

        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class UpscalerPipeline(StableDiffusionPipeline):
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
        ],
        low_res_scheduler: Union[
            DDIMScheduler,
            DDPMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            SharkEulerDiscreteScheduler,
            DEISMultistepScheduler,
        ],
        sd_model: SharkifyStableDiffusionModel,
        import_mlir: bool,
        use_lora: str,
        ondemand: bool,
    ):
        super().__init__(scheduler, sd_model, import_mlir, use_lora, ondemand)
        self.low_res_scheduler = low_res_scheduler

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(self, latents, use_base_vae, cpu_scheduling):
        latents = 1 / 0.08333 * (latents.float())
        latents_numpy = latents
        if cpu_scheduling:
            latents_numpy = latents.detach().numpy()

        profile_device = start_profiling(file_path="vae.rdc")
        vae_start = time.time()
        images = self.vae("forward", (latents_numpy,))
        vae_inf_time = (time.time() - vae_start) * 1000
        end_profiling(profile_device)
        self.log += f"\nVAE Inference time (ms): {vae_inf_time:.3f}"

        images = torch.from_numpy(images)
        images = (images.detach().cpu() * 255.0).numpy()
        images = images.round()

        images = torch.from_numpy(images).to(torch.uint8).permute(0, 2, 3, 1)
        pil_images = [Image.fromarray(image) for image in images.numpy()]
        return pil_images

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
                height,
                width,
            ),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.is_scale_input_called = True
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def produce_img_latents(
        self,
        latents,
        image,
        text_embeddings,
        guidance_scale,
        noise_level,
        total_timesteps,
        dtype,
        cpu_scheduling,
        extra_step_kwargs,
        return_all_latents=False,
    ):
        step_time_sum = 0
        latent_history = [latents]
        text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
        text_embeddings_numpy = text_embeddings.detach().numpy()
        self.load_unet()
        for i, t in tqdm(enumerate(total_timesteps)):
            step_start_time = time.time()
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )
            latent_model_input = torch.cat([latent_model_input, image], dim=1)
            timestep = torch.tensor([t]).to(dtype).detach().numpy()
            if cpu_scheduling:
                latent_model_input = latent_model_input.detach().numpy()

            # Profiling Unet.
            profile_device = start_profiling(file_path="unet.rdc")
            noise_pred = self.unet(
                "forward",
                (
                    latent_model_input,
                    timestep,
                    text_embeddings_numpy,
                    noise_level,
                ),
            )
            end_profiling(profile_device)
            noise_pred = torch.from_numpy(noise_pred)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            if cpu_scheduling:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
            else:
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )

            latent_history.append(latents)
            step_time = (time.time() - step_start_time) * 1000
            #  self.log += (
            #      f"\nstep = {i} | timestep = {t} | time = {step_time:.2f}ms"
            #  )
            step_time_sum += step_time

        if self.ondemand:
            self.unload_unet()
        avg_step_time = step_time_sum / len(total_timesteps)
        self.log += f"\nAverage step time: {avg_step_time}ms/it"

        if not return_all_latents:
            return latents
        all_latents = torch.cat(latent_history, dim=0)
        return all_latents

    def generate_images(
        self,
        prompts,
        neg_prompts,
        image,
        batch_size,
        height,
        width,
        num_inference_steps,
        noise_level,
        guidance_scale,
        seed,
        max_length,
        dtype,
        use_base_vae,
        cpu_scheduling,
    ):
        # prompts and negative prompts must be a list.
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(neg_prompts, str):
            neg_prompts = [neg_prompts]

        prompts = prompts * batch_size
        neg_prompts = neg_prompts * batch_size

        # seed generator to create the inital latent noise. Also handle out of range seeds.
        # TODO: Wouldn't it be preferable to just report an error instead of modifying the seed on the fly?
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        if seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)
        generator = torch.manual_seed(seed)

        # Get text embeddings with weight emphasis from prompts
        text_embeddings = self.encode_prompts_weight(prompts, neg_prompts)

        # 4. Preprocess image
        image = preprocess(image).to(dtype)

        # 5. Add noise to image
        noise_level = torch.tensor([noise_level], dtype=torch.long)
        noise = torch.randn(
            image.shape,
            generator=generator,
        ).to(dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        image = torch.cat([image] * 2)
        noise_level = torch.cat([noise_level] * image.shape[0])

        height, width = image.shape[2:]
        # Get initial latents
        init_latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            dtype=dtype,
        )

        eta = 0.0
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # guidance scale as a float32 tensor.
        #  guidance_scale = torch.tensor(guidance_scale).to(torch.float32)

        # Get Image latents
        latents = self.produce_img_latents(
            latents=init_latents,
            image=image,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            total_timesteps=self.scheduler.timesteps,
            dtype=dtype,
            cpu_scheduling=cpu_scheduling,
            extra_step_kwargs=extra_step_kwargs,
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
