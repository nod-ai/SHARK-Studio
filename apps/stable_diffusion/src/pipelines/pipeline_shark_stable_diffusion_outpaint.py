import torch
from tqdm.auto import tqdm
import numpy as np
from random import randint
from PIL import Image, ImageDraw, ImageFilter
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
import math


class OutpaintPipeline(StableDiffusionPipeline):
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

    def prepare_mask_and_masked_image(self, image, mask, mask_blur):
        if mask_blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        # preprocess image
        if isinstance(image, (Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], Image.Image):
            mask = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask], axis=0
            )
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)

        return mask, masked_image

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
    ):
        mask = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8)
        )
        mask = mask.to(dtype)

        masked_image = masked_image.to(dtype)
        masked_image_latents = self.vae_encode("forward", (masked_image,))
        masked_image_latents = torch.from_numpy(masked_image_latents)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )
        return mask, masked_image_latents

    def get_matched_noise(
        self, _np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05
    ):
        # helper fft routines that keep ortho normalization and auto-shift before and after fft
        def _fft2(data):
            if data.ndim > 2:  # has channels
                out_fft = np.zeros(
                    (data.shape[0], data.shape[1], data.shape[2]),
                    dtype=np.complex128,
                )
                for c in range(data.shape[2]):
                    c_data = data[:, :, c]
                    out_fft[:, :, c] = np.fft.fft2(
                        np.fft.fftshift(c_data), norm="ortho"
                    )
                    out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
            else:  # one channel
                out_fft = np.zeros(
                    (data.shape[0], data.shape[1]), dtype=np.complex128
                )
                out_fft[:, :] = np.fft.fft2(
                    np.fft.fftshift(data), norm="ortho"
                )
                out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

            return out_fft

        def _ifft2(data):
            if data.ndim > 2:  # has channels
                out_ifft = np.zeros(
                    (data.shape[0], data.shape[1], data.shape[2]),
                    dtype=np.complex128,
                )
                for c in range(data.shape[2]):
                    c_data = data[:, :, c]
                    out_ifft[:, :, c] = np.fft.ifft2(
                        np.fft.fftshift(c_data), norm="ortho"
                    )
                    out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
            else:  # one channel
                out_ifft = np.zeros(
                    (data.shape[0], data.shape[1]), dtype=np.complex128
                )
                out_ifft[:, :] = np.fft.ifft2(
                    np.fft.fftshift(data), norm="ortho"
                )
                out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

            return out_ifft

        def _get_gaussian_window(width, height, std=3.14, mode=0):
            window_scale_x = float(width / min(width, height))
            window_scale_y = float(height / min(width, height))

            window = np.zeros((width, height))
            x = (np.arange(width) / width * 2.0 - 1.0) * window_scale_x
            for y in range(height):
                fy = (y / height * 2.0 - 1.0) * window_scale_y
                if mode == 0:
                    window[:, y] = np.exp(-(x**2 + fy**2) * std)
                else:
                    window[:, y] = (
                        1 / ((x**2 + 1.0) * (fy**2 + 1.0))
                    ) ** (std / 3.14)

            return window

        def _get_masked_window_rgb(np_mask_grey, hardness=1.0):
            np_mask_rgb = np.zeros(
                (np_mask_grey.shape[0], np_mask_grey.shape[1], 3)
            )
            if hardness != 1.0:
                hardened = np_mask_grey[:] ** hardness
            else:
                hardened = np_mask_grey[:]
            for c in range(3):
                np_mask_rgb[:, :, c] = hardened[:]
            return np_mask_rgb

        def _match_cumulative_cdf(source, template):
            src_values, src_unique_indices, src_counts = np.unique(
                source.ravel(), return_inverse=True, return_counts=True
            )
            tmpl_values, tmpl_counts = np.unique(
                template.ravel(), return_counts=True
            )

            # calculate normalized quantiles for each array
            src_quantiles = np.cumsum(src_counts) / source.size
            tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

            interp_a_values = np.interp(
                src_quantiles, tmpl_quantiles, tmpl_values
            )
            return interp_a_values[src_unique_indices].reshape(source.shape)

        def _match_histograms(image, reference):
            if image.ndim != reference.ndim:
                raise ValueError(
                    "Image and reference must have the same number of channels."
                )

            if image.shape[-1] != reference.shape[-1]:
                raise ValueError(
                    "Number of channels in the input image and reference image must match!"
                )

            matched = np.empty(image.shape, dtype=image.dtype)
            for channel in range(image.shape[-1]):
                matched_channel = _match_cumulative_cdf(
                    image[..., channel], reference[..., channel]
                )
                matched[..., channel] = matched_channel

            matched = matched.astype(np.float64, copy=False)
            return matched

        width = _np_src_image.shape[0]
        height = _np_src_image.shape[1]
        num_channels = _np_src_image.shape[2]

        np_src_image = _np_src_image[:] * (1.0 - np_mask_rgb)
        np_mask_grey = np.sum(np_mask_rgb, axis=2) / 3.0
        img_mask = np_mask_grey > 1e-6
        ref_mask = np_mask_grey < 1e-3

        # rather than leave the masked area black, we get better results from fft by filling the average unmasked color
        windowed_image = _np_src_image * (
            1.0 - _get_masked_window_rgb(np_mask_grey)
        )
        windowed_image /= np.max(windowed_image)
        windowed_image += np.average(_np_src_image) * np_mask_rgb

        src_fft = _fft2(
            windowed_image
        )  # get feature statistics from masked src img
        src_dist = np.absolute(src_fft)
        src_phase = src_fft / src_dist

        # create a generator with a static seed to make outpainting deterministic / only follow global seed
        rng = np.random.default_rng(0)

        noise_window = _get_gaussian_window(
            width, height, mode=1
        )  # start with simple gaussian noise
        noise_rgb = rng.random((width, height, num_channels))
        noise_grey = np.sum(noise_rgb, axis=2) / 3.0
        # the colorfulness of the starting noise is blended to greyscale with a parameter
        noise_rgb *= color_variation
        for c in range(num_channels):
            noise_rgb[:, :, c] += (1.0 - color_variation) * noise_grey

        noise_fft = _fft2(noise_rgb)
        for c in range(num_channels):
            noise_fft[:, :, c] *= noise_window
        noise_rgb = np.real(_ifft2(noise_fft))
        shaped_noise_fft = _fft2(noise_rgb)
        shaped_noise_fft[:, :, :] = (
            np.absolute(shaped_noise_fft[:, :, :]) ** 2
            * (src_dist**noise_q)
            * src_phase
        )  # perform the actual shaping

        # color_variation
        brightness_variation = 0.0
        contrast_adjusted_np_src = (
            _np_src_image[:] * (brightness_variation + 1.0)
            - brightness_variation * 2.0
        )

        shaped_noise = np.real(_ifft2(shaped_noise_fft))
        shaped_noise -= np.min(shaped_noise)
        shaped_noise /= np.max(shaped_noise)
        shaped_noise[img_mask, :] = _match_histograms(
            shaped_noise[img_mask, :] ** 1.0,
            contrast_adjusted_np_src[ref_mask, :],
        )
        shaped_noise = (
            _np_src_image[:] * (1.0 - np_mask_rgb) + shaped_noise * np_mask_rgb
        )

        matched_noise = shaped_noise[:]

        return np.clip(matched_noise, 0.0, 1.0)

    def generate_images(
        self,
        prompts,
        neg_prompts,
        image,
        pixels,
        mask_blur,
        is_left,
        is_right,
        is_top,
        is_bottom,
        noise_q,
        color_variation,
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

        process_width = width
        process_height = height
        left = pixels if is_left else 0
        right = pixels if is_right else 0
        up = pixels if is_top else 0
        down = pixels if is_bottom else 0
        target_w = math.ceil((image.width + left + right) / 64) * 64
        target_h = math.ceil((image.height + up + down) / 64) * 64

        if left > 0:
            left = left * (target_w - image.width) // (left + right)
        if right > 0:
            right = target_w - image.width - left
        if up > 0:
            up = up * (target_h - image.height) // (up + down)
        if down > 0:
            down = target_h - image.height - up

        def expand(
            init_img,
            expand_pixels,
            is_left=False,
            is_right=False,
            is_top=False,
            is_bottom=False,
        ):
            is_horiz = is_left or is_right
            is_vert = is_top or is_bottom
            pixels_horiz = expand_pixels if is_horiz else 0
            pixels_vert = expand_pixels if is_vert else 0

            res_w = init_img.width + pixels_horiz
            res_h = init_img.height + pixels_vert
            process_res_w = math.ceil(res_w / 64) * 64
            process_res_h = math.ceil(res_h / 64) * 64

            img = Image.new("RGB", (process_res_w, process_res_h))
            img.paste(
                init_img,
                (pixels_horiz if is_left else 0, pixels_vert if is_top else 0),
            )

            msk = Image.new("RGB", (process_res_w, process_res_h), "white")
            draw = ImageDraw.Draw(msk)
            draw.rectangle(
                (
                    expand_pixels + mask_blur if is_left else 0,
                    expand_pixels + mask_blur if is_top else 0,
                    msk.width - expand_pixels - mask_blur
                    if is_right
                    else res_w,
                    msk.height - expand_pixels - mask_blur
                    if is_bottom
                    else res_h,
                ),
                fill="black",
            )

            np_image = (np.asarray(img) / 255.0).astype(np.float64)
            np_mask = (np.asarray(msk) / 255.0).astype(np.float64)
            noised = self.get_matched_noise(
                np_image, np_mask, noise_q, color_variation
            )
            output_image = Image.fromarray(
                np.clip(noised * 255.0, 0.0, 255.0).astype(np.uint8),
                mode="RGB",
            )

            target_width = (
                min(width, init_img.width + pixels_horiz)
                if is_horiz
                else img.width
            )
            target_height = (
                min(height, init_img.height + pixels_vert)
                if is_vert
                else img.height
            )
            crop_region = (
                0 if is_left else output_image.width - target_width,
                0 if is_top else output_image.height - target_height,
                target_width if is_left else output_image.width,
                target_height if is_top else output_image.height,
            )
            mask_to_process = msk.crop(crop_region)
            image_to_process = output_image.crop(crop_region)

            # Preprocess mask and image
            mask, masked_image = self.prepare_mask_and_masked_image(
                image_to_process, mask_to_process, mask_blur
            )

            # Prepare mask latent variables
            mask, masked_image_latents = self.prepare_mask_latents(
                mask=mask,
                masked_image=masked_image,
                batch_size=batch_size,
                height=height,
                width=width,
                dtype=dtype,
            )

            # Get Image latents
            latents = self.produce_img_latents(
                latents=init_latents,
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                total_timesteps=self.scheduler.timesteps,
                dtype=dtype,
                cpu_scheduling=cpu_scheduling,
                mask=mask,
                masked_image_latents=masked_image_latents,
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

            res_img = all_imgs[0].resize(
                (image_to_process.width, image_to_process.height)
            )
            output_image.paste(
                res_img,
                (
                    0 if is_left else output_image.width - res_img.width,
                    0 if is_top else output_image.height - res_img.height,
                ),
            )
            output_image = output_image.crop((0, 0, res_w, res_h))

            return output_image

        img = image.resize((width, height))
        if left > 0:
            img = expand(img, left, is_left=True)
        if right > 0:
            img = expand(img, right, is_right=True)
        if up > 0:
            img = expand(img, up, is_top=True)
        if down > 0:
            img = expand(img, down, is_bottom=True)

        return [img]
