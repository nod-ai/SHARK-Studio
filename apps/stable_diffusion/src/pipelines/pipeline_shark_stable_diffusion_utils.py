import torch
import numpy as np
from transformers import CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm
import time
from typing import Union
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
from shark.shark_inference import SharkInference
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.models import (
    SharkifyStableDiffusionModel,
    get_vae,
    get_clip,
    get_unet,
    get_tokenizer,
)
from apps.stable_diffusion.src.utils import (
    start_profiling,
    end_profiling,
)
import sys

SD_STATE_IDLE = "idle"
SD_STATE_CANCEL = "cancel"


class StableDiffusionPipeline:
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
        ],
        sd_model: SharkifyStableDiffusionModel,
        import_mlir: bool,
        use_lora: str,
        ondemand: bool,
    ):
        self.vae = None
        self.text_encoder = None
        self.unet = None
        self.tokenizer = get_tokenizer()
        self.scheduler = scheduler
        # TODO: Implement using logging python utility.
        self.log = ""
        self.status = SD_STATE_IDLE
        self.sd_model = sd_model
        self.import_mlir = import_mlir
        self.use_lora = use_lora
        self.ondemand = ondemand

    def load_clip(self):
        if self.text_encoder is not None:
            return

        if self.import_mlir or self.use_lora:
            if not self.import_mlir:
                print(
                    "Warning: LoRA provided but import_mlir not specified. Importing MLIR anyways."
                )
            self.text_encoder = self.sd_model.clip()
        else:
            try:
                self.text_encoder = get_clip()
            except:
                print("download pipeline failed, falling back to import_mlir")
                self.text_encoder = self.sd_model.clip()

    def unload_clip(self):
        del self.text_encoder
        self.text_encoder = None

    def load_unet(self):
        if self.unet is not None:
            return

        if self.import_mlir or self.use_lora:
            self.unet = self.sd_model.unet()
        else:
            try:
                self.unet = get_unet()
            except:
                print("download pipeline failed, falling back to import_mlir")
                self.unet = self.sd_model.unet()

    def unload_unet(self):
        del self.unet
        self.unet = None

    def load_vae(self):
        if self.vae is not None:
            return

        if self.import_mlir or self.use_lora:
            self.vae = self.sd_model.vae()
        else:
            try:
                self.vae = get_vae()
            except:
                print("download pipeline failed, falling back to import_mlir")
                self.vae = self.sd_model.vae()

    def unload_vae(self):
        del self.vae
        self.vae = None

    def encode_prompts(self, prompts, neg_prompts, max_length):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Get unconditional embeddings as well
        uncond_input = self.tokenizer(
            neg_prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input = torch.cat([uncond_input.input_ids, text_input.input_ids])

        self.load_clip()
        clip_inf_start = time.time()
        text_embeddings = self.text_encoder("forward", (text_input,))
        clip_inf_time = (time.time() - clip_inf_start) * 1000
        # self.unload_clip()
        self.log += f"\nClip Inference time (ms) = {clip_inf_time:.3f}"

        return text_embeddings

    def decode_latents(self, latents, use_base_vae, cpu_scheduling):
        if use_base_vae:
            latents = 1 / 0.18215 * latents

        latents_numpy = latents
        if cpu_scheduling:
            latents_numpy = latents.detach().numpy()

        profile_device = start_profiling(file_path="vae.rdc")
        vae_start = time.time()
        images = self.vae("forward", (latents_numpy,))
        vae_inf_time = (time.time() - vae_start) * 1000
        end_profiling(profile_device)
        self.log += f"\nVAE Inference time (ms): {vae_inf_time:.3f}"

        if use_base_vae:
            images = torch.from_numpy(images)
            images = (images.detach().cpu() * 255.0).numpy()
            images = images.round()

        images = torch.from_numpy(images).to(torch.uint8).permute(0, 2, 3, 1)
        pil_images = [Image.fromarray(image) for image in images.numpy()]
        return pil_images

    def produce_img_latents(
        self,
        latents,
        text_embeddings,
        guidance_scale,
        total_timesteps,
        dtype,
        cpu_scheduling,
        mask=None,
        masked_image_latents=None,
        return_all_latents=False,
    ):
        self.status = SD_STATE_IDLE
        step_time_sum = 0
        latent_history = [latents]
        text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
        text_embeddings_numpy = text_embeddings.detach().numpy()
        self.load_unet()
        for i, t in tqdm(enumerate(total_timesteps)):
            step_start_time = time.time()
            timestep = torch.tensor([t]).to(dtype).detach().numpy()
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

            # Profiling Unet.
            profile_device = start_profiling(file_path="unet.rdc")
            noise_pred = self.unet(
                "forward",
                (
                    latent_model_input,
                    timestep,
                    text_embeddings_numpy,
                    guidance_scale,
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

            if self.status == SD_STATE_CANCEL:
                break

        if self.ondemand:
            self.unload_unet()
        avg_step_time = step_time_sum / len(total_timesteps)
        self.log += f"\nAverage step time: {avg_step_time}ms/it"

        if not return_all_latents:
            return latents
        all_latents = torch.cat(latent_history, dim=0)
        return all_latents

    @classmethod
    def from_pretrained(
        cls,
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
        import_mlir: bool,
        model_id: str,
        ckpt_loc: str,
        custom_vae: str,
        precision: str,
        max_length: int,
        batch_size: int,
        height: int,
        width: int,
        use_base_vae: bool,
        use_tuned: bool,
        ondemand: bool,
        low_cpu_mem_usage: bool = False,
        debug: bool = False,
        use_stencil: str = None,
        use_lora: str = "",
        ddpm_scheduler: DDPMScheduler = None,
        use_quantize=None,
    ):
        if (
            not import_mlir
            and not use_lora
            and cls.__name__ == "StencilPipeline"
        ):
            sys.exit("StencilPipeline not supported with SharkTank currently.")

        is_inpaint = cls.__name__ in [
            "InpaintPipeline",
            "OutpaintPipeline",
        ]
        is_upscaler = cls.__name__ in ["UpscalerPipeline"]

        sd_model = SharkifyStableDiffusionModel(
            model_id,
            ckpt_loc,
            custom_vae,
            precision,
            max_len=max_length,
            batch_size=batch_size,
            height=height,
            width=width,
            use_base_vae=use_base_vae,
            use_tuned=use_tuned,
            low_cpu_mem_usage=low_cpu_mem_usage,
            debug=debug,
            is_inpaint=is_inpaint,
            is_upscaler=is_upscaler,
            use_stencil=use_stencil,
            use_lora=use_lora,
            use_quantize=use_quantize,
        )

        if cls.__name__ in ["UpscalerPipeline"]:
            return cls(
                scheduler,
                ddpm_scheduler,
                sd_model,
                import_mlir,
                use_lora,
                ondemand,
            )

        return cls(scheduler, sd_model, import_mlir, use_lora, ondemand)
