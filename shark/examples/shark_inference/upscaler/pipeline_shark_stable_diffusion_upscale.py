import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import torch

import PIL
from PIL import Image
from diffusers.utils import is_accelerate_available
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers import logging
from diffusers.pipeline_utils import ImagePipelineOutput
from opt_params import get_unet, get_vae, get_clip
from tqdm.auto import tqdm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(
            lambda x: x - x % 64, (w, h)
        )  # resize to integer multiple of 64

        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def shark_run_wrapper(model, *args):
    np_inputs = tuple([x.detach().numpy() for x in args])
    outputs = model("forward", np_inputs)
    return torch.from_numpy(outputs)


class SharkStableDiffusionUpscalePipeline:
    def __init__(
        self,
        model_id,
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.low_res_scheduler = DDPMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )
        self.vae = get_vae()
        self.unet = get_unet()
        self.text_encoder = get_clip()
        self.max_noise_level = (350,)
        self._execution_device = "cpu"

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[
            -1
        ] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        # if (
        # hasattr(self.text_encoder.config, "use_attention_mask")
        # and self.text_encoder.config.use_attention_mask
        # ):
        # attention_mask = text_inputs.attention_mask.to(device)
        # else:
        # attention_mask = None

        text_embeddings = shark_run_wrapper(
            self.text_encoder, text_input_ids.to(device)
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            # if (
            # hasattr(self.text_encoder.config, "use_attention_mask")
            # and self.text_encoder.config.use_attention_mask
            # ):
            # attention_mask = uncond_input.attention_mask.to(device)
            # else:
            # attention_mask = None

            uncond_embeddings = shark_run_wrapper(
                self.text_encoder,
                uncond_input.input_ids.to(device),
            )
            uncond_embeddings = uncond_embeddings

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1
            )
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents with 0.18215->0.08333
    def decode_latents(self, latents):
        latents = 1 / 0.08333 * latents
        image = shark_run_wrapper(self.vae, latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs(self, prompt, image, noise_level, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or `list` but is {type(image)}"
            )

        # verify batch size of prompt and image are same if image is a list or tensor
        if isinstance(image, list) or isinstance(image, torch.Tensor):
            if isinstance(prompt, str):
                batch_size = 1
            else:
                batch_size = len(prompt)
            if isinstance(image, list):
                image_batch_size = len(image)
            else:
                image_batch_size = image.shape[0]
            if batch_size != image_batch_size:
                raise ValueError(
                    f"`prompt` has batch size {batch_size} and `image` has batch size {image_batch_size}."
                    " Please make sure that passed `prompt` matches the batch size of `image`."
                )

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L") for image in images
            ]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            if device == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(
                    shape, generator=generator, device="cpu", dtype=dtype
                ).to(device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=device, dtype=dtype
                )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[
            torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]
        ],
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[
            Callable[[int, int, torch.FloatTensor], None]
        ] = None,
        callback_steps: Optional[int] = 1,
    ):
        # 1. Check inputs
        self.check_inputs(prompt, image, noise_level, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Preprocess image
        image = preprocess(image)
        image = image.to(dtype=text_embeddings.dtype, device=device)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Add noise to image
        noise_level = torch.tensor(
            [noise_level], dtype=torch.long, device=device
        )
        if device == "mps":
            # randn does not work reproducibly on mps
            noise = torch.randn(
                image.shape,
                generator=generator,
                device="cpu",
                dtype=text_embeddings.dtype,
            ).to(device)
        else:
            noise = torch.randn(
                image.shape,
                generator=generator,
                device=device,
                dtype=text_embeddings.dtype,
            )
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)

        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = torch.cat([image] * batch_multiplier * num_images_per_prompt)
        noise_level = torch.cat([noise_level] * image.shape[0])

        # 6. Prepare latent variables
        height, width = image.shape[2:]
        # num_channels_latents = self.vae.config.latent_channels
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7. Check that sizes of image and latents match
        num_channels_image = image.shape[1]
        # if (
        # num_channels_latents + num_channels_image
        # != self.unet.config.in_channels
        # ):
        # raise ValueError(
        # f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
        # f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
        # f" `num_channels_image`: {num_channels_image} "
        # f" = {num_channels_latents+num_channels_image}. Please verify the config of"
        # " `pipeline.unet` or your `image` input."
        # )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.scheduler.order
        )
        for i, t in tqdm(enumerate(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )
            latent_model_input = torch.cat([latent_model_input, image], dim=1)

            timestep = torch.tensor([t]).to(torch.float32)

            # predict the noise residual
            noise_pred = shark_run_wrapper(
                self.unet,
                latent_model_input.half(),
                timestep,
                text_embeddings.half(),
                noise_level,
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

            # # call the callback, if provided
            # if i == len(timesteps) - 1 or (
            # (i + 1) > num_warmup_steps
            # and (i + 1) % self.scheduler.order == 0
            # ):
            # progress_bar.update()
            # if callback is not None and i % callback_steps == 0:
            # callback(i, t, latents)

        # 10. Post-processing
        # make sure the VAE is in float32 mode, as it overflows in float16
        # self.vae.to(dtype=torch.float32)
        image = self.decode_latents(latents.float())

        # 11. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
