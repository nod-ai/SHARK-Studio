import torch
from tqdm.auto import tqdm
import numpy as np
from random import randint
from transformers import CLIPTokenizer
from typing import Union
from shark.shark_inference import SharkInference
from diffusers import (
    DDIMScheduler,
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

import cv2
from PIL import Image


class Text2ImagePipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: SharkInference,
        text_encoder: SharkInference,
        tokenizer: CLIPTokenizer,
        unet: SharkInference,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            KDPM2DiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            SharkEulerDiscreteScheduler,
            DEISMultistepScheduler,
        ],
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)

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

    # TODO: Move it to a separate ControlNet based specific model functions.
    def HWC3(self, x):
        assert x.dtype == np.uint8
        if x.ndim == 2:
            x = x[:, :, None]
        assert x.ndim == 3
        H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return np.concatenate([x, x, x], axis=2)
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)
            return y

    # TODO: Move it to a separate ControlNet based specific model functions.
    def resize_image(self, input_image, resolution):
        H, W, C = input_image.shape
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        img = cv2.resize(
            input_image,
            (W, H),
            interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
        )
        return img

    # TODO: Move it to a separate ControlNet based specific model functions.
    def hint_canny(
        self,
        image: Image.Image,
        width=512,
        height=512,
        low_threshold=100,
        high_threshold=200,
    ):
        with torch.no_grad():
            input_image = np.array(image)
            image_resolution = width

            img = self.resize_image(self.HWC3(input_image), image_resolution)

            class CannyDetector:
                def __call__(self, img, low_threshold, high_threshold):
                    return cv2.Canny(img, low_threshold, high_threshold)

            canny = CannyDetector()
            detected_map = canny(img, low_threshold, high_threshold)
            detected_map = self.HWC3(detected_map)
            return detected_map

    def generate_images(
        self,
        prompts,
        neg_prompts,
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
        # Control Embedding check & conversion
        # TODO: 1. `controlnet_hint`.
        #       2. Change `num_images_per_prompt`.
        #       3. Supply `controlnet_img`.
        from PIL import Image

        controlnet_img = Image.open("/home/abhishek/cn_1.png")
        controlnet_hint = self.hint_canny(controlnet_img)
        controlnet_hint = self.controlnet_hint_conversion(
            controlnet_hint, height, width, num_images_per_prompt=1
        )

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

        # Get initial latents
        init_latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            dtype=dtype,
        )

        # Get text embeddings from prompts
        text_embeddings = self.encode_prompts(prompts, neg_prompts, max_length)

        # guidance scale as a float32 tensor.
        guidance_scale = torch.tensor(guidance_scale).to(torch.float32)

        # Get Image latents
        latents = self.produce_img_latents(
            latents=init_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            total_timesteps=self.scheduler.timesteps,
            dtype=dtype,
            cpu_scheduling=cpu_scheduling,
            controlnet_hint=controlnet_hint,
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
