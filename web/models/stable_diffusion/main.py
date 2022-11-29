import os
import time

from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
from tqdm.auto import tqdm
import numpy as np
from numpy import iinfo
from random import randint
from models.stable_diffusion.opt_params import get_unet, get_vae, get_clip
from models.stable_diffusion.arguments import (
    args,
    schedulers,
    cache_obj,
    output_dir,
)


def stable_diff_inf(
    prompt: str,
    scheduler: str,
    iteration_count: int,
    batch_size: int,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    seed: int,
    precision: str,
    device: str,
    cache: bool,
    iree_vulkan_target_triple: str,
    live_preview: bool,
    save_img: bool,
    import_mlir: bool,
):

    start = time.time()
    # set seed value
    uint32_info = iinfo(np.uint32)
    if seed < uint32_info.min and seed >= uint32_info.max:
        seed = randint(uint32_info.min, uint32_info.max)

    args.set_params(
        prompt,
        scheduler,
        iteration_count,
        batch_size,
        steps,
        guidance,
        height,
        width,
        seed,
        precision,
        device,
        cache,
        iree_vulkan_target_triple,
        live_preview,
        save_img,
        import_mlir,
    )

    dtype = torch.float32 if args.precision == "fp32" else torch.half
    generator = torch.manual_seed(
        args.seed
    )  # Seed generator to create the inital latent noise

    scheduler_obj = schedulers[args.scheduler]
    guidance_scale = torch.tensor(args.guidance).to(torch.float32)

    # Initialize vae and unet models.
    vae, unet, clip = None, None, None
    is_model_initialized = False
    if args.cache and args.device == "vulkan" and not args.import_mlir:
        vae_key = f"vae_{args.precision}_vulkan"
        unet_key = f"unet_{args.precision}_vulkan"
        clip_key = "clip_vulkan"
        cached_keys = cache_obj.keys()
        if (
            vae_key in cached_keys
            and unet_key in cached_keys
            and clip_key in cached_keys
        ):
            vae, unet, clip = (
                cache_obj[vae_key],
                cache_obj[unet_key],
                cache_obj[clip_key],
            )
            is_model_initialized = True
    if not is_model_initialized:
        vae, unet, clip = get_vae(args), get_unet(args), get_clip(args)

    tokenizer = cache_obj["tokenizer"]

    text_input = tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_embeddings = clip.forward((text_input.input_ids,))
    text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = clip.forward((uncond_input.input_ids,))
    uncond_embeddings = torch.from_numpy(uncond_embeddings).to(dtype)

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (args.batch_size, 4, args.height // 8, args.width // 8),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)

    scheduler_obj.set_timesteps(args.steps)
    scheduler_obj.is_scale_input_called = True

    latents = latents * scheduler.init_noise_sigma
    text_embeddings_numpy = text_embeddings.detach().numpy()

    avg_ms = 0
    out_img = None
    text_output = ""
    for i, t in tqdm(enumerate(scheduler_obj.timesteps)):

        text_output += f"\n Iteration = {i} | Timestep = {t} | "
        step_start = time.time()
        timestep = torch.tensor([t]).to(dtype).detach().numpy()
        latents_model_input = scheduler.scale_model_input(latents, t)
        latents_numpy = latents_model_input.detach().numpy()

        noise_pred = unet.forward(
            (
                latents_numpy,
                timestep,
                text_embeddings_numpy,
                guidance_scale,
            )
        )
        noise_pred = torch.from_numpy(noise_pred)
        step_time = time.time() - step_start
        avg_ms += step_time
        step_ms = int((step_time) * 1000)
        text_output += f"Time = {step_ms}ms."
        print(f" \nIteration = {i}, Time = {step_ms}ms")
        latents = scheduler_obj.step(noise_pred, t, latents)["prev_sample"]

        if live_preview and i % 5 == 0:
            scaled_latents = 1 / 0.18215 * latents
            latents_numpy = scaled_latents.detach().numpy()
            image = vae.forward((latents_numpy,))
            image = torch.from_numpy(image)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            out_img = pil_images[0]
            yield out_img, text_output

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    latents_numpy = latents.detach().numpy()
    image = vae.forward((latents_numpy,))
    image = torch.from_numpy(image)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    out_img = pil_images[0]

    avg_ms = 1000 * avg_ms / args.steps
    text_output += f"\n\nAverage step time: {avg_ms}ms/it"
    print(f"\n\nAverage step time: {avg_ms}ms/it")

    total_time = time.time() - start
    text_output += f"\n\nTotal image generation time: {total_time}sec"

    if args.save_img:
        # save outputs.
        output_loc = f"{output_dir}/{time.time()}_{int(args.steps)}_{args.precision}_{args.device}.jpg"
        out_img.save(os.path.join(output_loc))
    yield out_img, text_output
