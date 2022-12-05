import torch
from PIL import Image
from tqdm.auto import tqdm
from models.stable_diffusion.cache_objects import cache_obj
from models.stable_diffusion.stable_args import args
import time


def set_ui_params(prompt, steps, guidance, seed):
    args.prompt = [prompt]
    args.steps = steps
    args.guidance = guidance
    args.seed = seed


def stable_diff_inf(
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
):

    start = time.time()
    set_ui_params(prompt, steps, guidance, seed)
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    generator = torch.manual_seed(
        args.seed
    )  # Seed generator to create the inital latent noise
    guidance_scale = torch.tensor(args.guidance).to(torch.float32)
    # Initialize vae and unet models.
    vae, unet, clip, tokenizer, scheduler = (
        cache_obj["vae"],
        cache_obj["unet"],
        cache_obj["clip"],
        cache_obj["tokenizer"],
        cache_obj["scheduler"],
    )

    text_input = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_embeddings = clip.forward((text_input.input_ids,))
    text_embeddings = torch.from_numpy(text_embeddings).to(dtype)
    max_length = text_input.input_ids.shape[-1]

    uncond_input = tokenizer(
        [""],
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = clip.forward((uncond_input.input_ids,))
    uncond_embeddings = torch.from_numpy(uncond_embeddings).to(dtype)

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (1, 4, args.height // 8, args.width // 8),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)

    scheduler.set_timesteps(args.steps)
    scheduler.is_scale_input_called = True

    latents = latents * scheduler.init_noise_sigma
    text_embeddings_numpy = text_embeddings.detach().numpy()

    avg_ms = 0
    out_img = None
    for i, t in tqdm(enumerate(scheduler.timesteps)):

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
        print(f" \nIteration = {i}, Time = {step_ms}ms")
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

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
    total_time = time.time() - start

    text_output = f"prompt={args.prompt}"
    text_output += f"\nsteps={args.steps}, guidance_scale={args.guidance}, seed={args.seed}"
    text_output += f"\nAverage step time: {avg_ms}ms/it"
    print(f"\nAverage step time: {avg_ms}ms/it")
    text_output += f"\nTotal image generation time: {total_time}sec"
    print(f"\nTotal image generation time: {total_time}sec")

    return out_img, text_output
