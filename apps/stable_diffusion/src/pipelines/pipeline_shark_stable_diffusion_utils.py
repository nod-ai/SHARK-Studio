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
        ],
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        # TODO: Implement using logging python utility.
        self.log = ""
        # TODO: Make this dynamic like other models which'll be passed to StableDiffusionPipeline.
        from diffusers import UNet2DConditionModel

        self.controlnet = UNet2DConditionModel.from_pretrained(
            "/home/abhishek/weights/canny_weight", subfolder="controlnet"
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

    def produce_img_latents(
        self,
        latents,
        text_embeddings,
        guidance_scale,
        total_timesteps,
        dtype,
        cpu_scheduling,
        controlnet_hint=None,
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

            # Profiling Unet.
            profile_device = start_profiling(file_path="unet.rdc")
            if controlnet_hint is not None:
                if not torch.is_tensor(latent_model_input):
                    latent_model_input_1 = torch.from_numpy(
                        np.asarray(latent_model_input)
                    ).to(dtype)
                else:
                    latent_model_input_1 = latent_model_input
                control = self.controlnet(
                    latent_model_input_1,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                    controlnet_hint=controlnet_hint,
                )
                timestep = timestep.detach().numpy()
                # TODO: Pass `control` as it is to Unet. Same as TODO mentioned in model_wrappers.py.
                noise_pred = self.unet(
                    "forward",
                    (
                        latent_model_input,
                        timestep,
                        text_embeddings_numpy,
                        guidance_scale,
                        control[0],
                        control[1],
                        control[2],
                        control[3],
                        control[4],
                        control[5],
                        control[6],
                        control[7],
                        control[8],
                        control[9],
                        control[10],
                        control[11],
                        control[12],
                    ),
                    send_to_host=False,
                )
            else:
                timestep = timestep.detach().numpy()
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

    def controlnet_hint_conversion(
        self, controlnet_hint, height, width, num_images_per_prompt=1
    ):
        channels = 3
        if isinstance(controlnet_hint, torch.Tensor):
            # torch.Tensor: acceptble shape are any of chw, bchw(b==1) or bchw(b==num_images_per_prompt)
            shape_chw = (channels, height, width)
            shape_bchw = (1, channels, height, width)
            shape_nchw = (num_images_per_prompt, channels, height, width)
            if controlnet_hint.shape in [shape_chw, shape_bchw, shape_nchw]:
                controlnet_hint = controlnet_hint.to(
                    dtype=torch.float32, device=torch.device("cpu")
                )
                if controlnet_hint.shape != shape_nchw:
                    controlnet_hint = controlnet_hint.repeat(
                        num_images_per_prompt, 1, 1, 1
                    )
                return controlnet_hint
            else:
                raise ValueError(
                    f"Acceptble shape of `controlnet_hint` are any of ({channels}, {height}, {width}),"
                    + f" (1, {channels}, {height}, {width}) or ({num_images_per_prompt}, "
                    + f"{channels}, {height}, {width}) but is {controlnet_hint.shape}"
                )
        elif isinstance(controlnet_hint, np.ndarray):
            # np.ndarray: acceptable shape is any of hw, hwc, bhwc(b==1) or bhwc(b==num_images_per_promot)
            # hwc is opencv compatible image format. Color channel must be BGR Format.
            if controlnet_hint.shape == (height, width):
                controlnet_hint = np.repeat(
                    controlnet_hint[:, :, np.newaxis], channels, axis=2
                )  # hw -> hwc(c==3)
            shape_hwc = (height, width, channels)
            shape_bhwc = (1, height, width, channels)
            shape_nhwc = (num_images_per_prompt, height, width, channels)
            if controlnet_hint.shape in [shape_hwc, shape_bhwc, shape_nhwc]:
                controlnet_hint = torch.from_numpy(controlnet_hint.copy())
                controlnet_hint = controlnet_hint.to(
                    dtype=torch.float32, device=torch.device("cpu")
                )
                controlnet_hint /= 255.0
                if controlnet_hint.shape != shape_nhwc:
                    controlnet_hint = controlnet_hint.repeat(
                        num_images_per_prompt, 1, 1, 1
                    )
                controlnet_hint = controlnet_hint.permute(
                    0, 3, 1, 2
                )  # b h w c -> b c h w
                return controlnet_hint
            else:
                raise ValueError(
                    f"Acceptble shape of `controlnet_hint` are any of ({width}, {channels}), "
                    + f"({height}, {width}, {channels}), "
                    + f"(1, {height}, {width}, {channels}) or "
                    + f"({num_images_per_prompt}, {channels}, {height}, {width}) but is {controlnet_hint.shape}"
                )
        elif isinstance(controlnet_hint, Image.Image):
            if controlnet_hint.size == (width, height):
                controlnet_hint = controlnet_hint.convert(
                    "RGB"
                )  # make sure 3 channel RGB format
                controlnet_hint = np.array(controlnet_hint)  # to numpy
                controlnet_hint = controlnet_hint[:, :, ::-1]  # RGB -> BGR
                return self.controlnet_hint_conversion(
                    controlnet_hint, height, width, num_images_per_prompt
                )
            else:
                raise ValueError(
                    f"Acceptable image size of `controlnet_hint` is ({width}, {height}) but is {controlnet_hint.size}"
                )
        else:
            raise ValueError(
                f"Acceptable type of `controlnet_hint` are any of torch.Tensor, np.ndarray, PIL.Image.Image but is {type(controlnet_hint)}"
            )

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
    ):
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
            )
            if cls.__name__ in ["Image2ImagePipeline", "InpaintPipeline"]:
                clip, unet, vae, vae_encode = mlir_import()
                return cls(
                    vae_encode, vae, clip, get_tokenizer(), unet, scheduler
                )
            clip, unet, vae = mlir_import()
            return cls(vae, clip, get_tokenizer(), unet, scheduler)
        try:
            if cls.__name__ in ["Image2ImagePipeline", "InpaintPipeline"]:
                return cls(
                    get_vae_encode(),
                    get_vae(),
                    get_clip(),
                    get_tokenizer(),
                    get_unet(),
                    scheduler,
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
            )
            if cls.__name__ in ["Image2ImagePipeline", "InpaintPipeline"]:
                clip, unet, vae, vae_encode = mlir_import()
                return cls(
                    vae_encode, vae, clip, get_tokenizer(), unet, scheduler
                )
            clip, unet, vae = mlir_import()
            return cls(vae, clip, get_tokenizer(), unet, scheduler)
