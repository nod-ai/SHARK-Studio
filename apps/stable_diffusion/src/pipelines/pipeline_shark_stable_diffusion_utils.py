import torch
import numpy as np
from transformers import CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm
import time
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
)
from shark.shark_inference import SharkInference
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.models import (
    SharkifyStableDiffusionModel,
    get_vae_encode,
    get_vae,
    get_clip,
    get_unet,
    get_tokenizer,
)
from apps.stable_diffusion.src.utils import (
    start_profiling,
    end_profiling,
)


class StableDiffusionPipeline:
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
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        # TODO: Implement using logging python utility.
        self.log = ""

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

        clip_inf_start = time.time()
        text_embeddings = self.text_encoder("forward", (text_input,))
        clip_inf_time = (time.time() - clip_inf_start) * 1000
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

    def produce_stencil_latents(
        self,
        latents,
        text_embeddings,
        guidance_scale,
        total_timesteps,
        dtype,
        cpu_scheduling,
        controlnet_hint=None,
        controlnet=None,
        controlnet_conditioning_scale: float = 1.0,
        mask=None,
        masked_image_latents=None,
        return_all_latents=False,
    ):
        step_time_sum = 0
        latent_history = [latents]
        text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
        text_embeddings_numpy = text_embeddings.detach().numpy()
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
            control = controlnet(
                "forward",
                (
                    latent_model_input_1,
                    timestep,
                    text_embeddings,
                    controlnet_hint,
                ),
                send_to_host=False,
            )
            down_block_res_samples = control[0:12]
            mid_block_res_sample = control[12:]
            down_block_res_samples = [
                down_block_res_sample * controlnet_conditioning_scale
                for down_block_res_sample in down_block_res_samples
            ]
            mid_block_res_sample = (
                mid_block_res_sample[0] * controlnet_conditioning_scale
            )
            timestep = timestep.detach().numpy()
            # Profiling Unet.
            profile_device = start_profiling(file_path="unet.rdc")
            # TODO: Pass `control` as it is to Unet. Same as TODO mentioned in model_wrappers.py.
            noise_pred = self.unet(
                "forward",
                (
                    latent_model_input,
                    timestep,
                    text_embeddings_numpy,
                    guidance_scale,
                    down_block_res_samples[0],
                    down_block_res_samples[1],
                    down_block_res_samples[2],
                    down_block_res_samples[3],
                    down_block_res_samples[4],
                    down_block_res_samples[5],
                    down_block_res_samples[6],
                    down_block_res_samples[7],
                    down_block_res_samples[8],
                    down_block_res_samples[9],
                    down_block_res_samples[10],
                    down_block_res_samples[11],
                    mid_block_res_sample,
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

        avg_step_time = step_time_sum / len(total_timesteps)
        self.log += f"\nAverage step time: {avg_step_time}ms/it"

        if not return_all_latents:
            return latents
        all_latents = torch.cat(latent_history, dim=0)
        return all_latents

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
        step_time_sum = 0
        latent_history = [latents]
        text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
        text_embeddings_numpy = text_embeddings.detach().numpy()
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
        low_cpu_mem_usage: bool = False,
        use_stencil: str = None,
    ):
        is_inpaint = cls.__name__ in [
            "InpaintPipeline",
            "OutpaintPipeline",
        ]
        if import_mlir:
            mlir_import = SharkifyStableDiffusionModel(
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
                is_inpaint=is_inpaint,
                use_stencil=use_stencil,
            )
            if cls.__name__ in [
                "Image2ImagePipeline",
                "InpaintPipeline",
                "OutpaintPipeline",
            ]:
                clip, unet, vae, vae_encode = mlir_import()
                return cls(
                    vae_encode, vae, clip, get_tokenizer(), unet, scheduler
                )
            if cls.__name__ in ["StencilPipeline"]:
                clip, unet, vae, controlnet = mlir_import()
                return cls(
                    controlnet, vae, clip, get_tokenizer(), unet, scheduler
                )
            clip, unet, vae = mlir_import()
            return cls(vae, clip, get_tokenizer(), unet, scheduler)
        try:
            if cls.__name__ in [
                "Image2ImagePipeline",
                "InpaintPipeline",
                "OutpaintPipeline",
            ]:
                return cls(
                    get_vae_encode(),
                    get_vae(),
                    get_clip(),
                    get_tokenizer(),
                    get_unet(),
                    scheduler,
                )
            if cls.__name__ == "StencilPipeline":
                import sys

                sys.exit(
                    "StencilPipeline not supported with SharkTank currently."
                )
            return cls(
                get_vae(), get_clip(), get_tokenizer(), get_unet(), scheduler
            )
        except:
            print("download pipeline failed, falling back to import_mlir")
            mlir_import = SharkifyStableDiffusionModel(
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
                is_inpaint=is_inpaint,
            )
            if cls.__name__ in [
                "Image2ImagePipeline",
                "InpaintPipeline",
                "OutpaintPipeline",
            ]:
                clip, unet, vae, vae_encode = mlir_import()
                return cls(
                    vae_encode, vae, clip, get_tokenizer(), unet, scheduler
                )
            if cls.__name__ == "StencilPipeline":
                clip, unet, vae, controlnet = mlir_import()
                return cls(
                    controlnet, vae, clip, get_tokenizer(), unet, scheduler
                )
            clip, unet, vae = mlir_import()
            return cls(vae, clip, get_tokenizer(), unet, scheduler)
