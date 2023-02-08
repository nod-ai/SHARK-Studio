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
)
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
    StableDiffusionPipeline,
)


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
        ],
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)
        self.vae_encode = vae_encode

    def prepare_image_latents(
        self,
        image,
        batch_size,
        height,
        width,
        generator,
        num_inference_steps,
        dtype,
    ):
        # Pre process image -> get image encoded -> process latents

        # TODO: process with variable HxW combos

        # Pre process image
        image = image.resize((height, width))  # Current support for 512x512
        image_arr = np.stack([np.array(i) for i in (image,)], axis=0)
        image_arr = image_arr / 255.0
        image_arr = torch.from_numpy(image_arr).permute(0, 3, 1, 2).to(dtype)
        image_arr = 2 * (image_arr - 0.5)

        # image encode
        latents = self.encode_image((image_arr,))
        latents = torch.from_numpy(latents).to(dtype)

        # set scheduler steps
        self.scheduler.set_timesteps(num_inference_steps)

        # add noise to data
        latents = latents * self.scheduler.init_noise_sigma

        return latents

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
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        if seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)
        generator = torch.manual_seed(seed)

        # Get text embeddings from prompts
        text_embeddings = self.encode_prompts(prompts, neg_prompts, max_length)

        # guidance scale as a float32 tensor.
        guidance_scale = torch.tensor(guidance_scale).to(torch.float32)

        # Prepare input image latent
        image_latents = self.prepare_image_latents(
            image=image,
            batch_size=batch_size,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            dtype=dtype,
        )

        # Get Image latents
        latents = self.produce_img_latents(
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            total_timesteps=self.scheduler.timesteps,
            dtype=dtype,
            cpu_scheduling=cpu_scheduling,
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
