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
    DPMSolverSinglestepScheduler,
    KDPM2AncestralDiscreteScheduler,
    HeunDiscreteScheduler,
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
import gc
from typing import List, Optional

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
            DDPMScheduler,
            DPMSolverSinglestepScheduler,
            KDPM2AncestralDiscreteScheduler,
            HeunDiscreteScheduler,
        ],
        sd_model: SharkifyStableDiffusionModel,
        import_mlir: bool,
        use_lora: str,
        ondemand: bool,
        is_f32_vae: bool = False,
    ):
        self.vae = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.unet = None
        self.unet_512 = None
        self.model_max_length = 77
        self.scheduler = scheduler
        # TODO: Implement using logging python utility.
        self.log = ""
        self.status = SD_STATE_IDLE
        self.sd_model = sd_model
        self.import_mlir = import_mlir
        self.use_lora = use_lora
        self.ondemand = ondemand
        self.is_f32_vae = is_f32_vae
        # TODO: Find a better workaround for fetching base_model_id early
        #  enough for CLIPTokenizer.
        try:
            self.tokenizer = get_tokenizer()
        except:
            self.load_unet()
            self.unload_unet()
            self.tokenizer = get_tokenizer()

    def load_clip(self):
        if self.text_encoder is not None:
            return

        if self.import_mlir or self.use_lora:
            if not self.import_mlir:
                print(
                    "Warning: LoRA provided but import_mlir not specified. "
                    "Importing MLIR anyways."
                )
            self.text_encoder = self.sd_model.clip()
        else:
            try:
                self.text_encoder = get_clip()
            except Exception as e:
                print(e)
                print("download pipeline failed, falling back to import_mlir")
                self.text_encoder = self.sd_model.clip()

    def unload_clip(self):
        del self.text_encoder
        self.text_encoder = None

    def load_clip_sdxl(self):
        if self.text_encoder and self.text_encoder_2:
            return

        if self.import_mlir or self.use_lora:
            if not self.import_mlir:
                print(
                    "Warning: LoRA provided but import_mlir not specified. "
                    "Importing MLIR anyways."
                )
            self.text_encoder, self.text_encoder_2 = self.sd_model.sdxl_clip()
        else:
            try:
                # TODO: Fix this for SDXL
                self.text_encoder = get_clip()
            except Exception as e:
                print(e)
                print("download pipeline failed, falling back to import_mlir")
                (
                    self.text_encoder,
                    self.text_encoder_2,
                ) = self.sd_model.sdxl_clip()

    def unload_clip_sdxl(self):
        del self.text_encoder, self.text_encoder_2
        self.text_encoder = None
        self.text_encoder_2 = None

    def load_unet(self):
        if self.unet is not None:
            return

        if self.import_mlir or self.use_lora:
            self.unet = self.sd_model.unet()
        else:
            try:
                self.unet = get_unet()
            except Exception as e:
                print(e)
                print("download pipeline failed, falling back to import_mlir")
                self.unet = self.sd_model.unet()

    def unload_unet(self):
        del self.unet
        self.unet = None

    def load_unet_512(self):
        if self.unet_512 is not None:
            return

        if self.import_mlir or self.use_lora:
            self.unet_512 = self.sd_model.unet(use_large=True)
        else:
            try:
                self.unet_512 = get_unet(use_large=True)
            except Exception as e:
                print(e)
                print("download pipeline failed, falling back to import_mlir")
                self.unet_512 = self.sd_model.unet(use_large=True)

    def unload_unet_512(self):
        del self.unet_512
        self.unet_512 = None

    def load_vae(self):
        if self.vae is not None:
            return

        if self.import_mlir or self.use_lora:
            self.vae = self.sd_model.vae()
        else:
            try:
                self.vae = get_vae()
            except Exception as e:
                print(e)
                print("download pipeline failed, falling back to import_mlir")
                self.vae = self.sd_model.vae()

    def unload_vae(self):
        del self.vae
        self.vae = None
        gc.collect()

    def encode_prompt_sdxl(
        self,
        prompt: str,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        self.tokenizer_2 = get_tokenizer("tokenizer_2")
        self.load_clip_sdxl()
        tokenizers = (
            [self.tokenizer, self.tokenizer_2]
            if self.tokenizer is not None
            else [self.tokenizer_2]
        )
        text_encoders = (
            [self.text_encoder, self.text_encoder_2]
            if self.text_encoder is not None
            else [self.text_encoder_2]
        )

        # textual inversion: procecss multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt]
        for prompt, tokenizer, text_encoder in zip(
            prompts, tokenizers, text_encoders
        ):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            text_encoder_output = text_encoder("forward", (text_input_ids,))
            prompt_embeds = torch.from_numpy(text_encoder_output[0])
            pooled_prompt_embeds = torch.from_numpy(text_encoder_output[1])

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = (
            negative_prompt is None
            and self.config.force_zeros_for_empty_prompt
        )
        if (
            do_classifier_free_guidance
            and negative_prompt_embeds is None
            and zero_out_negative_prompt
        ):
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(
                pooled_prompt_embeds
            )
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(
                negative_prompt
            ):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt, negative_prompt_2]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(
                uncond_tokens, tokenizers, text_encoders
            ):
                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_encoder_output = text_encoder(
                    "forward", (uncond_input.input_ids,)
                )
                negative_prompt_embeds = torch.from_numpy(
                    text_encoder_output[0]
                )
                negative_pooled_prompt_embeds = torch.from_numpy(
                    text_encoder_output[1]
                )

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(
                negative_prompt_embeds_list, dim=-1
            )

        if self.ondemand:
            self.unload_clip_sdxl()
            gc.collect()

        # TODO: Look into dtype for text_encoder_2!
        prompt_embeds = prompt_embeds.to(dtype=torch.float16)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=torch.float32)
        negative_prompt_embeds = negative_prompt_embeds.repeat(
            1, num_images_per_prompt, 1
        )
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            1, num_images_per_prompt
        ).view(bs_embed * num_images_per_prompt, -1)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
            1, num_images_per_prompt
        ).view(bs_embed * num_images_per_prompt, -1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

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
        if self.ondemand:
            self.unload_clip()
            gc.collect()
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
        if text_embeddings.shape[1] <= self.model_max_length:
            self.load_unet()
        else:
            self.load_unet_512()
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
            if text_embeddings.shape[1] <= self.model_max_length:
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
            else:
                noise_pred = self.unet_512(
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
            self.unload_unet_512()
            gc.collect()

        avg_step_time = step_time_sum / len(total_timesteps)
        self.log += f"\nAverage step time: {avg_step_time}ms/it"

        if not return_all_latents:
            return latents
        all_latents = torch.cat(latent_history, dim=0)
        return all_latents

    def produce_img_latents_sdxl(
        self,
        latents,
        total_timesteps,
        add_text_embeds,
        add_time_ids,
        prompt_embeds,
        cpu_scheduling,
        guidance_scale,
        dtype,
    ):
        # return None
        self.status = SD_STATE_IDLE
        step_time_sum = 0
        extra_step_kwargs = {"generator": None}
        self.load_unet()
        for i, t in tqdm(enumerate(total_timesteps)):
            step_start_time = time.time()
            timestep = torch.tensor([t]).to(dtype).detach().numpy()
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            ).to(dtype)

            noise_pred = self.unet(
                "forward",
                (
                    latent_model_input,
                    timestep,
                    prompt_embeds,
                    add_text_embeds,
                    add_time_ids,
                    guidance_scale,
                ),
                send_to_host=False,
            )
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]

            step_time = (time.time() - step_start_time) * 1000
            step_time_sum += step_time

            if self.status == SD_STATE_CANCEL:
                break
        if self.ondemand:
            self.unload_unet()
            gc.collect()

        avg_step_time = step_time_sum / len(total_timesteps)
        self.log += f"\nAverage step time: {avg_step_time}ms/it"

        return latents

    def decode_latents_sdxl(self, latents, is_fp32_vae):
        # latents are in unet dtype here so switch if we want to use fp32
        if is_fp32_vae:
            print("Casting latents to float32 for VAE")
            latents = latents.to(torch.float32)
        images = self.vae("forward", (latents,))
        images = (torch.from_numpy(images) / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image[:, :, :3]) for image in images]

        return pil_images

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
            DDPMScheduler,
            DPMSolverSinglestepScheduler,
            KDPM2AncestralDiscreteScheduler,
            HeunDiscreteScheduler,
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
        stencils: list[str] = [],
        # stencil_images: list[Image] = []
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
        is_sdxl = cls.__name__ in ["Text2ImageSDXLPipeline"]

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
            is_sdxl=is_sdxl,
            stencils=stencils,
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

        if cls.__name__ == "StencilPipeline":
            return cls(
                scheduler, sd_model, import_mlir, use_lora, ondemand, stencils
            )
        if cls.__name__ == "Text2ImageSDXLPipeline":
            is_fp32_vae = True if "16" not in custom_vae else False
            return cls(
                scheduler,
                sd_model,
                import_mlir,
                use_lora,
                ondemand,
                is_fp32_vae,
            )

        return cls(scheduler, sd_model, import_mlir, use_lora, ondemand)

    # #####################################################
    # Implements text embeddings with weights from prompts
    # https://huggingface.co/AlanB/lpw_stable_diffusion_mod
    # #####################################################
    def encode_prompts_weight(
        self,
        prompt,
        negative_prompt,
        model_max_length,
        do_classifier_free_guidance=True,
        max_embeddings_multiples=1,
        num_images_per_prompt=1,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation.
                Ignored when not using guidance
                (i.e., ignored if `guidance_scale` is less than `1`).
            model_max_length (int):
                SHARK: pass the max length instead of relying on
                pipe.tokenizer.model_max_length
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not,
                SHARK: must be set to True as we always expect neg embeddings
                (defaulted to True)
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the
                max output length of text encoder.
                SHARK: max_embeddings_multiples>1 produce a tensor shape error
                (defaulted to 1)
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
                SHARK: num_images_per_prompt is not used (defaulted to 1)
        """

        # SHARK: Save model_max_length, load the clip and init inference time
        self.model_max_length = model_max_length
        self.load_clip()
        clip_inf_start = time.time()

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: "
                f"{negative_prompt} has batch size {len(negative_prompt)}, "
                f"but `prompt`: {prompt} has batch size {batch_size}. "
                f"Please make sure that passed `negative_prompt` matches "
                "the batch size of `prompt`."
            )

        text_embeddings, uncond_embeddings = get_weighted_text_embeddings(
            pipe=self,
            prompt=prompt,
            uncond_prompt=negative_prompt
            if do_classifier_free_guidance
            else None,
            max_embeddings_multiples=max_embeddings_multiples,
        )
        # SHARK: we are not using num_images_per_prompt
        # bs_embed, seq_len, _ = text_embeddings.shape
        # text_embeddings = text_embeddings.repeat(
        #     1,
        #     num_images_per_prompt,
        #     1
        # )
        # text_embeddings = (
        #     text_embeddings.view(
        #         bs_embed * num_images_per_prompt,
        #         seq_len,
        #         -1
        #     )
        # )

        if do_classifier_free_guidance:
            # SHARK: we are not using num_images_per_prompt
            # bs_embed, seq_len, _ = uncond_embeddings.shape
            # uncond_embeddings = (
            #     uncond_embeddings.repeat(
            #         1,
            #         num_images_per_prompt,
            #         1
            #     )
            # )
            # uncond_embeddings = (
            #     uncond_embeddings.view(
            #         bs_embed * num_images_per_prompt,
            #         seq_len,
            #         -1
            #     )
            # )
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        if text_embeddings.shape[1] > model_max_length:
            pad = (0, 0) * (len(text_embeddings.shape) - 2)
            pad = pad + (0, 512 - text_embeddings.shape[1])
            text_embeddings = torch.nn.functional.pad(text_embeddings, pad)

        # SHARK: Report clip inference time
        clip_inf_time = (time.time() - clip_inf_start) * 1000
        if self.ondemand:
            self.unload_clip()
            gc.collect()
        self.log += f"\nClip Inference time (ms) = {clip_inf_time:.3f}"

        return text_embeddings.numpy()


from typing import List, Optional, Union
import re

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs:
        text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(
    pipe: StableDiffusionPipeline, prompt: List[str], max_length: int
):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        print(
            "Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples"
        )
    return tokens, weights


def pad_tokens_and_weights(
    tokens,
    weights,
    max_length,
    bos,
    eos,
    no_boseos_middle=True,
    chunk_length=77,
):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = (
        max_length
        if no_boseos_middle
        else max_embeddings_multiples * chunk_length
    )
    for i in range(len(tokens)):
        tokens[i] = (
            [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        )
        if no_boseos_middle:
            weights[i] = (
                [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
            )
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][
                        j
                        * (chunk_length - 2) : min(
                            len(weights[i]), (j + 1) * (chunk_length - 2)
                        )
                    ]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe: StableDiffusionPipeline,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[
                :, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2
            ].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            # text_embedding = pipe.text_encoder(text_input_chunk)[0]
            # SHARK: deplicate the text_input as Shark runner expects tokens and neg tokens
            formatted_text_input_chunk = torch.cat(
                [text_input_chunk, text_input_chunk]
            )
            text_embedding = pipe.text_encoder(
                "forward", (formatted_text_input_chunk,)
            )[0]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        # SHARK: Convert the result to tensor
        # text_embeddings = torch.concat(text_embeddings, axis=1)
        text_embeddings_np = np.concatenate(np.array(text_embeddings))
        text_embeddings = torch.from_numpy(text_embeddings_np)[None, :]
    else:
        # SHARK: deplicate the text_input as Shark runner expects tokens and neg tokens
        # Convert the result to tensor
        # text_embeddings = pipe.text_encoder(text_input)[0]
        formatted_text_input = torch.cat([text_input, text_input])
        text_embeddings = pipe.text_encoder(
            "forward", (formatted_text_input,)
        )[0]
        text_embeddings = torch.from_numpy(text_embeddings)[None, :]
    return text_embeddings


# This function deals with NoneType values occuring in tokens after padding
# It switches out None with 49407 as truncating None values causes matrix dimension errors,
def filter_nonetype_tokens(tokens: List[List]):
    return [[49407 if token is None else token for token in tokens[0]]]


def get_weighted_text_embeddings(
    pipe: StableDiffusionPipeline,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
    Args:
        pipe (`StableDiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(
            pipe, prompt, max_length - 2
        )
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(
                pipe, uncond_prompt, max_length - 2
            )
    else:
        prompt_tokens = [
            token[1:-1]
            for token in pipe.tokenizer(
                prompt, max_length=max_length, truncation=True
            ).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1]
                for token in pipe.tokenizer(
                    uncond_prompt, max_length=max_length, truncation=True
                ).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(
            max_length, max([len(token) for token in uncond_tokens])
        )

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.model_max_length,
    )

    # FIXME: This is a hacky fix caused by tokenizer padding with None values
    prompt_tokens = filter_nonetype_tokens(prompt_tokens)

    # prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu")
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.model_max_length,
        )

        # FIXME: This is a hacky fix caused by tokenizer padding with None values
        uncond_tokens = filter_nonetype_tokens(uncond_tokens)

        # uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)
        uncond_tokens = torch.tensor(
            uncond_tokens, dtype=torch.long, device="cpu"
        )

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.model_max_length,
        no_boseos_middle=no_boseos_middle,
    )
    # prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
    prompt_weights = torch.tensor(
        prompt_weights, dtype=torch.float, device="cpu"
    )
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        # uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)
        uncond_weights = torch.tensor(
            uncond_weights, dtype=torch.float, device="cpu"
        )

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = (
            text_embeddings.float()
            .mean(axis=[-2, -1])
            .to(text_embeddings.dtype)
        )
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = (
            text_embeddings.float()
            .mean(axis=[-2, -1])
            .to(text_embeddings.dtype)
        )
        text_embeddings *= (
            (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        )
        if uncond_prompt is not None:
            previous_mean = (
                uncond_embeddings.float()
                .mean(axis=[-2, -1])
                .to(uncond_embeddings.dtype)
            )
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = (
                uncond_embeddings.float()
                .mean(axis=[-2, -1])
                .to(uncond_embeddings.dtype)
            )
            uncond_embeddings *= (
                (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
            )

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None
