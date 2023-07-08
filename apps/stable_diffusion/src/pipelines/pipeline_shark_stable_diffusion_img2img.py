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
from apps.stable_diffusion.src.models import (
    SharkifyStableDiffusionModel,
    get_vae_encode,
)


class Image2ImagePipeline(StableDiffusionPipeline):
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
        ondemand: bool,
    ):
        super().__init__(scheduler, sd_model, import_mlir, use_lora, ondemand)
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

        # TODO: process with variable HxW combos

        # Pre process image
        image = image.resize((width, height))
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
        use_stencil,
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

        # Get text embeddings with weight emphasis from prompts
        text_embeddings = self.encode_prompts_weight(
            prompts,
            neg_prompts,
            max_length,
            max_embeddings_multiples=max_embeddings_multiples,
        )

        # guidance scale as a float32 tensor.
        guidance_scale = torch.tensor(guidance_scale).to(torch.float32)

        # Prepare input image latent
        image_latents, final_timesteps = self.prepare_image_latents(
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
            latents=image_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            total_timesteps=final_timesteps,
            dtype=dtype,
            cpu_scheduling=cpu_scheduling,
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
