import torch
import numpy as np
from random import randint
from typing import Union
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    DDPMScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
)
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
    StableDiffusionPipeline,
)
from apps.stable_diffusion.src.models import SharkifyStableDiffusionModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Text2ImageSDXLPipeline(StableDiffusionPipeline):
    def __init__(
        self,
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
            DDPMScheduler,
            DPMSolverSinglestepScheduler,
            KDPM2AncestralDiscreteScheduler,
            HeunDiscreteScheduler,
        ],
        sd_model: SharkifyStableDiffusionModel,
        import_mlir: bool,
        use_lora: str,
        ondemand: bool,
    ):
        super().__init__(scheduler, sd_model, import_mlir, use_lora, ondemand)

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

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype
    ):
        add_time_ids = list(
            original_size + crops_coords_top_left + target_size
        )

        # self.unet.config.addition_time_embed_dim IS 256.
        # self.text_encoder_2.config.projection_dim IS 1280.
        passed_add_embed_dim = 256 * len(add_time_ids) + 1280
        expected_add_embed_dim = 2816
        # self.unet.add_embedding.linear_1.in_features IS 2816.

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

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
        max_embeddings_multiples,
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

        # Get initial latents.
        init_latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            dtype=dtype,
        )

        # Get text embeddings.
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt_sdxl(
            prompt=prompts,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=neg_prompts,
        )

        # Prepare timesteps.
        self.scheduler.set_timesteps(num_inference_steps)

        timesteps = self.scheduler.timesteps

        # Prepare added time ids & embeddings.
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
        )

        prompt_embeds = torch.cat(
            [negative_prompt_embeds, prompt_embeds], dim=0
        )
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds
        add_text_embeds = add_text_embeds.to(dtype)
        add_time_ids = add_time_ids.repeat(batch_size * 1, 1)

        # guidance scale as a float32 tensor.
        guidance_scale = torch.tensor(guidance_scale).to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        add_time_ids = add_time_ids.to(dtype)

        # Get Image latents.
        latents = self.produce_img_latents_sdxl(
            init_latents,
            timesteps,
            add_text_embeds,
            add_time_ids,
            prompt_embeds,
            cpu_scheduling,
            guidance_scale,
            dtype,
        )

        # Img latents -> PIL images.
        all_imgs = []
        self.load_vae()
        # imgs = self.decode_latents_sdxl(None)
        # all_imgs.extend(imgs)
        for i in range(0, latents.shape[0], batch_size):
            imgs = self.decode_latents_sdxl(latents[i : i + batch_size])
            all_imgs.extend(imgs)
        if self.ondemand:
            self.unload_vae()

        return all_imgs
