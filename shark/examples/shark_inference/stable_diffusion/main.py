import os
import sys

if "AMD_ENABLE_LLPC" not in os.environ:
    os.environ["AMD_ENABLE_LLPC"] = "1"

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"

from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
)
from tqdm.auto import tqdm
import numpy as np
from random import randint
from stable_args import args
from datetime import datetime as dt
import json
import re
from pathlib import Path
from model_wrappers import SharkifyStableDiffusionModel

# This has to come before importing cache objects
if args.clear_all:
    print("CLEARING ALL, EXPECT SEVERAL MINUTES TO RECOMPILE")
    from glob import glob
    import shutil

    vmfbs = glob(os.path.join(os.getcwd(), "*.vmfb"))
    for vmfb in vmfbs:
        if os.path.exists(vmfb):
            os.remove(vmfb)
    # Temporary workaround of deleting yaml files to incorporate diffusers' pipeline.
    # TODO: Remove this once we have better weight updation logic.
    inference_yaml = ["v2-inference-v.yaml", "v1-inference.yaml"]
    for yaml in inference_yaml:
        if os.path.exists(yaml):
            os.remove(yaml)
    home = os.path.expanduser("~")
    if os.name == "nt":  # Windows
        appdata = os.getenv("LOCALAPPDATA")
        shutil.rmtree(os.path.join(appdata, "AMD/VkCache"), ignore_errors=True)
        shutil.rmtree(os.path.join(home, "shark_tank"), ignore_errors=True)
    elif os.name == "unix":
        shutil.rmtree(os.path.join(home, ".cache/AMD/VkCache"))
        shutil.rmtree(os.path.join(home, ".local/shark_tank"))


from utils import set_init_device_flags, disk_space_check, preprocessCKPT

from schedulers import (
    SharkEulerDiscreteScheduler,
)
import time
from shark.iree_utils.compile_utils import dump_isas


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


if __name__ == "__main__":
    dtype = torch.float32 if args.precision == "fp32" else torch.half

    # Make it as default prompt
    if len(args.prompts) == 0:
        args.prompts = ["cyberpunk forest by Salvador Dali"]

    prompt = args.prompts
    neg_prompt = args.negative_prompts
    height = args.height
    width = args.width
    num_inference_steps = args.steps  # Number of denoising steps

    # Scale for classifier-free guidance
    guidance_scale = torch.tensor(args.guidance_scale).to(torch.float32)

    batch_size = args.batch_size
    prompt = prompt * batch_size if len(prompt) == 1 else prompt
    len_of_prompt = len(prompt)
    assert (
        len_of_prompt == batch_size
    ), f"no. of prompts ({len_of_prompt}) is not equal to batch_size ({batch_size})"
    print("Running StableDiffusion with the following config :-")
    print(f"Batch size : {batch_size}")
    print(f"Prompts : {prompt}")
    print(f"Runs : {args.runs}")

    # Try to make neg_prompt equal to batch_size by appending blank strings.
    for i in range(batch_size - len(neg_prompt)):
        neg_prompt.append("")

    set_init_device_flags()
    disk_space_check(Path.cwd())

    if not args.import_mlir:
        from opt_params import get_unet, get_vae, get_clip

        clip = get_clip()
        unet = get_unet()
        vae = get_vae()
    else:
        if args.ckpt_loc != "":
            assert args.ckpt_loc.lower().endswith(
                (".ckpt", ".safetensors")
            ), "checkpoint files supported can be any of [.ckpt, .safetensors] type"
            preprocessCKPT()
        mlir_import = SharkifyStableDiffusionModel(
            args.hf_model_id,
            args.ckpt_loc,
            args.precision,
            max_len=args.max_length,
            batch_size=batch_size,
            height=height,
            width=width,
            use_base_vae=args.use_base_vae,
            use_tuned=args.use_tuned,
        )
        clip, unet, vae = mlir_import()

    if args.dump_isa:
        dump_isas(args.dispatch_benchmarks_dir)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="scheduler",
    )
    cpu_scheduling = True
    if args.hf_model_id == "stabilityai/stable-diffusion-2-1":
        tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="tokenizer"
        )

        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            subfolder="scheduler",
        )

    if args.hf_model_id == "stabilityai/stable-diffusion-2-1-base":
        tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer"
        )

        if args.use_compiled_scheduler:
            scheduler = SharkEulerDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base",
                subfolder="scheduler",
            )
            scheduler.compile()
            cpu_scheduling = False
        else:
            scheduler = EulerDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base",
                subfolder="scheduler",
            )
    for run in range(args.runs):
        # Handle out of range seeds.
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        seed = args.seed
        if run >= 1 or seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)
        generator = torch.manual_seed(
            seed
        )  # Seed generator to create the inital latent noise

        # create a random initial latent.
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype)
        if run == 0:
            # Warmup phase to improve performance.
            if args.warmup_count >= 1:
                vae_warmup_input = torch.clone(latents).detach().numpy()
                clip_warmup_input = torch.randint(1, 2, (2, args.max_length))
            for i in range(args.warmup_count):
                vae("forward", (vae_warmup_input,))
                clip("forward", (clip_warmup_input,))

        start = time.time()
        if run == 0:
            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
            )
            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer(
                neg_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input = torch.cat(
                [uncond_input.input_ids, text_input.input_ids]
            )

            clip_inf_start = time.time()
            text_embeddings = clip("forward", (text_input,))
            clip_inf_end = time.time()
            text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
            text_embeddings_numpy = text_embeddings.detach().numpy()

            scheduler.set_timesteps(num_inference_steps)
            scheduler.is_scale_input_called = True

        latents = latents * scheduler.init_noise_sigma

        avg_ms = 0
        for i, t in tqdm(
            enumerate(scheduler.timesteps), disable=args.hide_steps
        ):
            step_start = time.time()
            if not args.hide_steps:
                print(f"i = {i} t = {t}", end="")
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
                    guidance_scale,
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
                print(f" ({step_ms}ms)")

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

        print(f"\nStats for run {run}:")
        print(f"Average step time: {avg_ms}ms/it")
        print(f"Clip Inference time (ms) = {clip_inf_time:.3f}")
        print(f"VAE Inference time (ms): {vae_inf_time:.3f}")
        print(f"\nTotal image generation time: {total_time}sec")

        images = torch.from_numpy(images).to(torch.uint8).permute(0, 2, 3, 1)
        pil_images = [Image.fromarray(image) for image in images.numpy()]

        if args.output_dir is not None:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path.cwd()
        disk_space_check(output_path, lim=5)
        for i in range(batch_size):
            json_store = {
                "prompt": prompt[i],
                "negative prompt": args.negative_prompts[i],
                "seed": seed,
                "hf_model_id": args.hf_model_id,
                "precision": args.precision,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "scheduler": args.scheduler,
            }
            prompt_slice = re.sub("[^a-zA-Z0-9]", "_", prompt[i][:15])
            img_name = f"{prompt_slice}_{seed}_{run}_{i}_{dt.now().strftime('%y%m%d_%H%M%S')}"
            if args.output_img_format == "jpg":
                pil_images[i].save(
                    output_path / f"{img_name}.jpg",
                    quality=95,
                    subsampling=0,
                    optimize=True,
                    progressive=True,
                )
            else:
                pil_images[i].save(output_path / f"{img_name}.png", "PNG")
                if args.output_img_format not in ["png", "jpg"]:
                    print(
                        f"[ERROR] Format {args.output_img_format} is not supported yet."
                        "saving image as png. Supported formats png / jpg"
                    )
            with open(output_path / f"{img_name}.json", "w") as f:
                f.write(json.dumps(json_store, indent=4))
