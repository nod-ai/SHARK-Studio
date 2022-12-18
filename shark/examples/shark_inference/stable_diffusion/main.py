import os

os.environ["AMD_ENABLE_LLPC"] = "1"

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
from stable_args import args
from utils import get_shark_model, set_iree_runtime_flags
from opt_params import get_unet, get_vae, get_clip
from schedulers import (
    SharkEulerDiscreteScheduler,
)
import time
import sys
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

    prompt = args.prompts
    neg_prompt = args.negative_prompts
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    if args.version == "v2.1":
        height = 768
        width = 768

    num_inference_steps = args.steps  # Number of denoising steps

    # Scale for classifier-free guidance
    guidance_scale = torch.tensor(args.guidance_scale).to(torch.float32)

    generator = torch.manual_seed(
        args.seed
    )  # Seed generator to create the inital latent noise

    # TODO: Add support for batch_size > 1.
    batch_size = len(prompt)
    if batch_size != 1:
        sys.exit("More than one prompt is not supported yet.")
    if batch_size != len(neg_prompt):
        sys.exit("prompts and negative prompts must be of same length")

    set_iree_runtime_flags()
    unet = get_unet()
    vae = get_vae()
    clip = get_clip()
    if args.dump_isa:
        dump_isas(args.dispatch_benchmarks_dir)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="scheduler",
    )
    cpu_scheduling = True
    if args.version == "v2.1":
        tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1", subfolder="tokenizer"
        )

        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            subfolder="scheduler",
        )

    if args.version == "v2.1base":
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

    start = time.time()

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        return_tensors="pt",
    )

    clip_inf_start = time.time()
    text_embeddings = clip.forward((text_input.input_ids,))
    clip_inf_end = time.time()
    text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        neg_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_clip_inf_start = time.time()
    uncond_embeddings = clip.forward((uncond_input.input_ids,))
    uncond_clip_inf_end = time.time()
    uncond_embeddings = torch.from_numpy(uncond_embeddings).to(dtype)

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, 4, height // 8, width // 8),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)

    scheduler.set_timesteps(num_inference_steps)
    scheduler.is_scale_input_called = True

    latents = latents * scheduler.init_noise_sigma
    text_embeddings_numpy = text_embeddings.detach().numpy()
    avg_ms = 0

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        step_start = time.time()
        print(f"i = {i} t = {t}", end="")
        timestep = torch.tensor([t]).to(dtype).detach().numpy()
        latent_model_input = scheduler.scale_model_input(latents, t)
        if cpu_scheduling:
            latent_model_input = latent_model_input.detach().numpy()

        profile_device = start_profiling(file_path="unet.rdc")

        noise_pred = unet.forward(
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
        print(f" ({step_ms}ms)")

    avg_ms = 1000 * avg_ms / args.steps
    print(f"Average step time: {avg_ms}ms/it")

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    # latents = latents.
    latents_numpy = latents
    if cpu_scheduling:
        latents_numpy = latents.detach().numpy()
    profile_device = start_profiling(file_path="vae.rdc")
    vae_start = time.time()
    image = vae.forward((latents_numpy,))
    vae_end = time.time()
    end_profiling(profile_device)
    image = torch.from_numpy(image)
    image = image.detach().cpu().permute(0, 2, 3, 1) * 255.0
    images = image.numpy().round().astype("uint8")
    total_end = time.time()

    clip_inf_time = (clip_inf_end - clip_inf_start) * 1000
    uncond_clip_inf_time = (uncond_clip_inf_end - uncond_clip_inf_start) * 1000
    avg_clip_inf = (clip_inf_time + uncond_clip_inf_time) / 2
    vae_inf_time = (vae_end - vae_start) * 1000
    print(
        f"Clip Inference Avg time (ms) = ({clip_inf_time:.3f} + {uncond_clip_inf_time:.3f}) / 2 = {avg_clip_inf:.3f}"
    )
    print(f"VAE Inference time (ms): {vae_inf_time:.3f}")
    print(f"Total image generation runtime (s): {total_end - start:.4f}")

    pil_images = [Image.fromarray(image) for image in images]
    for i in range(batch_size):
        pil_images[i].save(f"{args.prompts[i]}_{i}.jpg")
