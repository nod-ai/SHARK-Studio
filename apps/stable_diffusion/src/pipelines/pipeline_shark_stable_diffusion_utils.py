import torch
from transformers import CLIPTokenizer
import torchvision.transforms as T
from tqdm.auto import tqdm
import time
from typing import Union
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from shark.shark_inference import SharkInference
from ..schedulers import SharkEulerDiscreteScheduler
from ..models import (
    SharkifyStableDiffusionModel,
    get_vae,
    get_clip,
    get_unet,
    get_tokenizer,
)
from ..utils import start_profiling, end_profiling, preprocessCKPT


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
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            SharkEulerDiscreteScheduler,
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

        transform = T.ToPILImage()
        pil_images = [
            transform(image)
            for image in torch.from_numpy(images).to(torch.uint8)
        ]
        return pil_images

    def produce_img_latents(
        self,
        latents,
        text_embeddings,
        guidance_scale,
        total_timesteps,
        dtype,
        cpu_scheduling,
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
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            SharkEulerDiscreteScheduler,
        ],
        import_mlir: bool,
        model_id: str,
        ckpt_loc: str,
        precision: str,
        max_length: int,
        batch_size: int,
        height: int,
        width: int,
        use_base_vae: bool,
    ):
        init_kwargs = None
        if import_mlir:
            if ckpt_loc:
                preprocessCKPT()
            mlir_import = SharkifyStableDiffusionModel(
                model_id,
                ckpt_loc,
                precision,
                max_len=max_length,
                batch_size=batch_size,
                height=height,
                width=width,
                use_base_vae=use_base_vae,
            )
            clip, unet, vae = mlir_import()
            return cls(vae, clip, get_tokenizer(), unet, scheduler)
        return cls(
            get_vae(), get_clip(), get_tokenizer(), get_unet(), scheduler
        )
