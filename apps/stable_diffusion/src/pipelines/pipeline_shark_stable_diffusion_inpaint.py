import torch
from tqdm.auto import tqdm
import numpy as np
from random import randint
from PIL import Image, ImageOps
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
)
from apps.stable_diffusion.src.schedulers import SharkEulerDiscreteScheduler
from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
    StableDiffusionPipeline,
)
from apps.stable_diffusion.src.models import (
    SharkifyStableDiffusionModel,
    get_vae_encode,
)


class InpaintPipeline(StableDiffusionPipeline):
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
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_crop_region(self, mask, pad=0):
        h, w = mask.shape

        crop_left = 0
        for i in range(w):
            if not (mask[:, i] == 0).all():
                break
            crop_left += 1

        crop_right = 0
        for i in reversed(range(w)):
            if not (mask[:, i] == 0).all():
                break
            crop_right += 1

        crop_top = 0
        for i in range(h):
            if not (mask[i] == 0).all():
                break
            crop_top += 1

        crop_bottom = 0
        for i in reversed(range(h)):
            if not (mask[i] == 0).all():
                break
            crop_bottom += 1

        return (
            int(max(crop_left - pad, 0)),
            int(max(crop_top - pad, 0)),
            int(min(w - crop_right + pad, w)),
            int(min(h - crop_bottom + pad, h)),
        )

    def expand_crop_region(
        self,
        crop_region,
        processing_width,
        processing_height,
        image_width,
        image_height,
    ):
        x1, y1, x2, y2 = crop_region

        ratio_crop_region = (x2 - x1) / (y2 - y1)
        ratio_processing = processing_width / processing_height

        if ratio_crop_region > ratio_processing:
            desired_height = (x2 - x1) / ratio_processing
            desired_height_diff = int(desired_height - (y2 - y1))
            y1 -= desired_height_diff // 2
            y2 += desired_height_diff - desired_height_diff // 2
            if y2 >= image_height:
                diff = y2 - image_height
                y2 -= diff
                y1 -= diff
            if y1 < 0:
                y2 -= y1
                y1 -= y1
            if y2 >= image_height:
                y2 = image_height
        else:
            desired_width = (y2 - y1) * ratio_processing
            desired_width_diff = int(desired_width - (x2 - x1))
            x1 -= desired_width_diff // 2
            x2 += desired_width_diff - desired_width_diff // 2
            if x2 >= image_width:
                diff = x2 - image_width
                x2 -= diff
                x1 -= diff
            if x1 < 0:
                x2 -= x1
                x1 -= x1
            if x2 >= image_width:
                x2 = image_width

        return x1, y1, x2, y2

    def resize_image(self, resize_mode, im, width, height):
        """
        resize_mode:
            0: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            1: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        """

        if resize_mode == 0:
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = (
                width if ratio > src_ratio else im.width * height // im.height
            )
            src_h = (
                height if ratio <= src_ratio else im.height * width // im.width
            )

            resized = im.resize((src_w, src_h), resample=Image.LANCZOS)
            res = Image.new("RGB", (width, height))
            res.paste(
                resized,
                box=(width // 2 - src_w // 2, height // 2 - src_h // 2),
            )

        else:
            ratio = width / height
            src_ratio = im.width / im.height

            src_w = (
                width if ratio < src_ratio else im.width * height // im.height
            )
            src_h = (
                height if ratio >= src_ratio else im.height * width // im.width
            )

            resized = im.resize((src_w, src_h), resample=Image.LANCZOS)
            res = Image.new("RGB", (width, height))
            res.paste(
                resized,
                box=(width // 2 - src_w // 2, height // 2 - src_h // 2),
            )

            if ratio < src_ratio:
                fill_height = height // 2 - src_h // 2
                res.paste(
                    resized.resize((width, fill_height), box=(0, 0, width, 0)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (width, fill_height),
                        box=(0, resized.height, width, resized.height),
                    ),
                    box=(0, fill_height + src_h),
                )
            elif ratio > src_ratio:
                fill_width = width // 2 - src_w // 2
                res.paste(
                    resized.resize(
                        (fill_width, height), box=(0, 0, 0, height)
                    ),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (fill_width, height),
                        box=(resized.width, 0, resized.width, height),
                    ),
                    box=(fill_width + src_w, 0),
                )

        return res

    def prepare_mask_and_masked_image(
        self,
        image,
        mask,
        height,
        width,
        inpaint_full_res,
        inpaint_full_res_padding,
    ):
        # preprocess image
        image = image.resize((width, height))
        mask = mask.resize((width, height))

        paste_to = ()
        overlay_image = None
        if inpaint_full_res:
            # prepare overlay image
            overlay_image = Image.new("RGB", (image.width, image.height))
            overlay_image.paste(
                image.convert("RGB"),
                mask=ImageOps.invert(mask.convert("L")),
            )

            # prepare mask
            mask = mask.convert("L")
            crop_region = self.get_crop_region(
                np.array(mask), inpaint_full_res_padding
            )
            crop_region = self.expand_crop_region(
                crop_region, width, height, mask.width, mask.height
            )
            x1, y1, x2, y2 = crop_region
            mask = mask.crop(crop_region)
            mask = self.resize_image(1, mask, width, height)
            paste_to = (x1, y1, x2 - x1, y2 - y1)

            # prepare image
            image = image.crop(crop_region)
            image = self.resize_image(1, image, width, height)

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

        return mask, masked_image, paste_to, overlay_image

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

        self.load_vae_encode()
        masked_image = masked_image.to(dtype)
        masked_image_latents = self.vae_encode("forward", (masked_image,))
        masked_image_latents = torch.from_numpy(masked_image_latents)
        if self.ondemand:
            self.unload_vae_encode()

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

    def apply_overlay(self, image, paste_loc, overlay):
        x, y, w, h = paste_loc
        image = self.resize_image(0, image, w, h)
        overlay.paste(image, (x, y))

        return overlay

    def generate_images(
        self,
        prompts,
        neg_prompts,
        image,
        mask_image,
        batch_size,
        height,
        width,
        inpaint_full_res,
        inpaint_full_res_padding,
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

        # Preprocess mask and image
        (
            mask,
            masked_image,
            paste_to,
            overlay_image,
        ) = self.prepare_mask_and_masked_image(
            image,
            mask_image,
            height,
            width,
            inpaint_full_res,
            inpaint_full_res_padding,
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

        if inpaint_full_res:
            output_image = self.apply_overlay(
                all_imgs[0], paste_to, overlay_image
            )
            return [output_image]

        return all_imgs
