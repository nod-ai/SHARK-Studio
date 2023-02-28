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
)
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
    StableDiffusionPipeline,
)
from apps.stable_diffusion.src.utils import controlnet_hint_conversion


class Image2ImagePipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae_encode: SharkInference,
        vae: SharkInference,
        text_encoder: SharkInference,
        tokenizer: CLIPTokenizer,
        unet: SharkInference,
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
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)
        self.vae_encode = vae_encode

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

    def preprocess_img(
        self,
        input_image,
        height,
        width,
        dtype,
    ):
        # TODO: process with variable HxW combos
        # Pre process image
        # expecting single image as input_image for stencil generation
        if type(input_image) == torch.Tensor:
            return input_image

        # if input image is of type PIL.Image.Image
        image_rs = np.resize(input_image, (width, height))
        image_arr = np.stack([np.array(i) for i in (image_rs,)], axis=0)
        image_arr = image_arr / 255.0
        image_arr = torch.from_numpy(image_arr).permute(0, 3, 1, 2).to(dtype)
        image_t = 2 * (image_arr - 0.5)

        return image_t

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
    ):
        # Pre process image -> get image encoded -> process latents
        image_arr = self.preprocess_img(image, height, width, dtype)

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

    def encode_image(self, input_image):
        vae_encode_start = time.time()
        latents = self.vae_encode("forward", input_image)
        vae_inf_time = (time.time() - vae_encode_start) * 1000
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
        use_stencil,
    ):
        # Control Embedding check & conversion
        # TODO: 1. Change `num_images_per_prompt`.
        stencil_unprocessed = controlnet_hint_conversion(
            image, use_stencil, height, width, dtype, num_images_per_prompt=1
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

        # Get text embeddings from prompts
        text_embeddings = self.encode_prompts(prompts, neg_prompts, max_length)

        # guidance scale as a float32 tensor.
        guidance_scale = torch.tensor(guidance_scale).to(torch.float32)

        # Prepare initial latent.
        init_latents = None
        final_timesteps = None
        stencil_t = None
        if use_stencil is not None:
            stencil_t = self.preprocess_img(
                stencil_unprocessed, height, width, dtype
            )
            init_latents = self.prepare_latents(
                batch_size=batch_size,
                height=height,
                width=width,
                generator=generator,
                num_inference_steps=num_inference_steps,
                dtype=dtype,
            )
            final_timesteps = self.scheduler.timesteps
        else:
            init_latents, final_timesteps = self.prepare_image_latents(
                image=image,
                batch_size=batch_size,
                height=height,
                width=width,
                generator=generator,
                num_inference_steps=num_inference_steps,
                strength=strength,
                dtype=dtype,
            )

        # Get Image latents
        latents = self.produce_img_latents(
            latents=init_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            total_timesteps=final_timesteps,
            dtype=dtype,
            cpu_scheduling=cpu_scheduling,
            stencil=stencil_t,
        )

        # Img latents -> PIL images
        all_imgs = []
        for i in tqdm(range(0, latents.shape[0], batch_size)):
            imgs = self.decode_latents(
                latents=latents[i : i + batch_size],
                use_base_vae=use_base_vae,
                cpu_scheduling=cpu_scheduling,
            )
            all_imgs.extend(imgs)

        return all_imgs
