import torch
import os
from PIL import Image
from tqdm.auto import tqdm
from models.stable_diffusion.cache_objects import model_cache
from models.stable_diffusion.stable_args import args
from models.stable_diffusion.utils import disk_space_check
from random import randint
import numpy as np
import time
import sys
from datetime import datetime as dt
from csv import DictWriter
import re
from pathlib import Path


if args.clear_all:
    print("CLEARING ALL, EXPECT SEVERAL MINUTES TO RECOMPILE")
    from glob import glob
    import shutil

    vmfbs = glob(os.path.join(os.getcwd(), "*.vmfb"))
    for vmfb in vmfbs:
        if os.path.exists(vmfb):
            os.remove(vmfb)
    home = os.path.expanduser("~")
    if os.name == "nt":  # Windows
        appdata = os.getenv("LOCALAPPDATA")
        shutil.rmtree(os.path.join(appdata, "AMD/VkCache"), ignore_errors=True)
        shutil.rmtree(os.path.join(home, "shark_tank"), ignore_errors=True)
    elif os.name == "unix":
        shutil.rmtree(os.path.join(home, ".cache/AMD/VkCache"))
        shutil.rmtree(os.path.join(home, ".local/shark_tank"))


# Helper function to profile the vulkan device.
def start_profiling(file_path="foo.rdc", profiling_mode="queue"):
    if args.vulkan_debug_utils and "vulkan" in args.device:
        import iree

        print(f"Profiling and saving to {file_path}.")
        vulkan_device = iree.runtime.get_device(args.device)
        vulkan_device.begin_profiling(mode=profiling_mode, file_path=file_path)
        return vulkan_device
    return None


def end_profiling(device):
    if device:
        return device.end_profiling()


def set_ui_params(
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    seed,
    scheduler_key,
    variant,
):
    args.prompts = [prompt]
    args.negative_prompts = [negative_prompt]
    args.steps = steps
    args.guidance_scale = torch.tensor(guidance_scale).to(torch.float32)
    args.seed = seed
    args.scheduler = scheduler_key
    args.variant = variant


# save output images and the inputs correspoding to it.
def save_output_img(output_img):
    output_path = args.output_dir if args.output_dir else Path.cwd()
    disk_space_check(output_path, lim=5)
    generated_imgs_path = Path(output_path, "generated_imgs")
    generated_imgs_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(generated_imgs_path, "imgs_history.csv")

    prompt_slice = re.sub("[^a-zA-Z0-9]", "_", args.prompts[0][:15])
    out_img_name = (
        f"{prompt_slice}_{args.seed}_{dt.now().strftime('%y%m%d_%H%M%S')}"
    )
    if args.output_img_format == "jpg":
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.jpg")
        output_img.save(
            out_img_path,
            quality=95,
            subsampling=0,
            optimize=True,
            progressive=True,
        )
    else:
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.png")
        output_img.save(out_img_path, "PNG")
        if args.output_img_format not in ["png", "jpg"]:
            print(
                f"[ERROR] Format {args.output_img_format} is not supported yet."
                "saving image as png. Supported formats png / jpg"
            )

    new_entry = {
        "VARIANT": args.variant,
        "VERSION": args.version,
        "SCHEDULER": args.scheduler,
        "PROMPT": args.prompts[0],
        "NEG_PROMPT": args.negative_prompts[0],
        "SEED": args.seed,
        "CFG_SCALE": float(args.guidance_scale),
        "PRECISION": args.precision,
        "STEPS": args.steps,
        "OUTPUT": out_img_path,
    }

    with open(csv_path, "a") as csv_obj:
        dictwriter_obj = DictWriter(csv_obj, fieldnames=list(new_entry.keys()))
        dictwriter_obj.writerow(new_entry)
        csv_obj.close()


def stable_diff_inf(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    scheduler_key: str,
    variant: str,
    device_key: str,
):
    # Handle out of range seeds.
    uint32_info = np.iinfo(np.uint32)
    uint32_min, uint32_max = uint32_info.min, uint32_info.max
    if seed < uint32_min or seed >= uint32_max:
        seed = randint(uint32_min, uint32_max)

    set_ui_params(
        prompt,
        negative_prompt,
        steps,
        guidance_scale,
        seed,
        scheduler_key,
        variant,
    )
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    generator = torch.manual_seed(
        args.seed
    )  # Seed generator to create the inital latent noise

    # set height and width.
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    if args.version == "v2_1":
        height = 768
        width = 768

    # get all cached data.
    disk_space_check(Path.cwd())
    model_cache.set_models(device_key)
    tokenizer = model_cache.tokenizer
    scheduler = model_cache.schedulers[args.scheduler]
    vae, unet, clip = model_cache.vae, model_cache.unet, model_cache.clip
    cpu_scheduling = not args.scheduler.startswith("Shark")

    # create a random initial latent.
    latents = torch.randn(
        (1, 4, height // 8, width // 8),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)

    # Warmup phase to improve performance.
    if args.warmup_count >= 1:
        vae_warmup_input = torch.clone(latents).detach().numpy()
        clip_warmup_input = torch.randint(1, 2, (2, args.max_length))
    for i in range(args.warmup_count):
        vae("forward", (vae_warmup_input,))
        clip("forward", (clip_warmup_input,))

    start = time.time()
    text_input = tokenizer(
        args.prompts,
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        return_tensors="pt",
    )
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        args.negative_prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input = torch.cat([uncond_input.input_ids, text_input.input_ids])

    clip_inf_start = time.time()
    text_embeddings = clip("forward", (text_input,))
    clip_inf_end = time.time()
    text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
    text_embeddings_numpy = text_embeddings.detach().numpy()

    scheduler.set_timesteps(args.steps)
    scheduler.is_scale_input_called = True

    latents = latents * scheduler.init_noise_sigma

    avg_ms = 0
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        step_start = time.time()
        timestep = torch.tensor([t]).to(dtype).detach().numpy()
        latent_model_input = scheduler.scale_model_input(latents, t)
        if cpu_scheduling:
            latent_model_input = latent_model_input.detach().numpy()

        profile_device = start_profiling(file_path="unet.rdc")
        noise_pred = unet(
            "forward",
            (
                latent_model_input,
                timestep,
                text_embeddings_numpy,
                args.guidance_scale,
            ),
            send_to_host=False,
        )
        end_profiling(profile_device)

        if cpu_scheduling:
            noise_pred = torch.from_numpy(noise_pred.to_host())
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        else:
            latents = scheduler.step(noise_pred, t, latents)
        step_time = time.time() - step_start
        avg_ms += step_time
        step_ms = int((step_time) * 1000)
        if not args.hide_steps:
            print(f" \nIteration = {i}, Time = {step_ms}ms")

    # scale and decode the image latents with vae
    if args.use_base_vae:
        latents = 1 / 0.18215 * latents
    latents_numpy = latents
    if cpu_scheduling:
        latents_numpy = latents.detach().numpy()
    profile_device = start_profiling(file_path="vae.rdc")
    vae_start = time.time()
    images = vae("forward", (latents_numpy,))
    vae_end = time.time()
    end_profiling(profile_device)
    if args.use_base_vae:
        image = torch.from_numpy(images)
        image = (image.detach().cpu() * 255.0).numpy()
        images = image.round()
    end_time = time.time()

    avg_ms = 1000 * avg_ms / args.steps
    clip_inf_time = (clip_inf_end - clip_inf_start) * 1000
    vae_inf_time = (vae_end - vae_start) * 1000
    total_time = end_time - start
    print(f"\nAverage step time: {avg_ms}ms/it")
    print(f"Clip Inference time (ms) = {clip_inf_time:.3f}")
    print(f"VAE Inference time (ms): {vae_inf_time:.3f}")
    print(f"\nTotal image generation time: {total_time}sec")

    # generate outputs to web.
    images = torch.from_numpy(images).to(torch.uint8).permute(0, 2, 3, 1)
    pil_images = [Image.fromarray(image) for image in images.numpy()]

    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nvariant={args.variant}, version={args.version}, scheduler={args.scheduler}"
    text_output += f"\ndevice={device_key}"
    text_output += f"\nsteps={args.steps}, guidance_scale={args.guidance_scale}, seed={args.seed}, size={height}x{width}"
    text_output += f"\nAverage step time: {avg_ms:.4f}ms/it"
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    save_output_img(pil_images[0])

    return pil_images[0], text_output
