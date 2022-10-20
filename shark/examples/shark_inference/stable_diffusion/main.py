from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
import numpy as np
from stable_args import args
from model_wrappers import (
    get_vae32,
    get_vae16,
    get_unet16_wrapped,
    get_unet32_wrapped,
)
from utils import get_shark_model

GCLOUD_BUCKET = "gs://shark_tank/prashant_nod"
VAE_FP16 = "vae_fp16"
VAE_FP32 = "vae_fp32"
UNET_FP16 = "unet_fp16"
UNET_FP32 = "unet_fp32"


def get_models():
    if args.precision == "fp16":
        if args.import_mlir == True:
            return get_unet16_wrapped(), get_vae16()
        else:
            return get_shark_model(GCLOUD_BUCKET, VAE_FP16), get_shark_model(
                GCLOUD_BUCKET, UNET_FP16
            )

    elif args.precision == "fp32":
        if args.import_mlir == True:
            return get_vae32(), get_unet32_wrapped()
        else:
            return get_shark_model(GCLOUD_BUCKET, VAE_FP32), get_shark_model(
                GCLOUD_BUCKET, UNET_FP32
            )


if __name__ == "__main__":

    dtype = torch.float32 if args.precision == "fp32" else torch.half

    prompt = [args.prompt]

    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion

    num_inference_steps = args.steps  # Number of denoising steps

    guidance_scale = 7.5  # Scale for classifier-free guidance

    generator = torch.manual_seed(
        42
    )  # Seed generator to create the inital latent noise

    batch_size = len(prompt)

    vae, unet = get_models()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_embeddings = text_encoder(text_input.input_ids)[0].to(dtype)
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0].to(dtype)

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, 4, height // 8, width // 8),
        generator=generator,
        dtype=dtype,
    )

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.sigmas[0]
    text_embeddings_numpy = text_embeddings.detach().numpy()

    for i, t in tqdm(enumerate(scheduler.timesteps)):

        print(f"i = {i} t = {t}")
        timestep = torch.tensor([t]).to(dtype).detach().numpy()
        latents_numpy = latents.detach().numpy()
        sigma_numpy = np.array(scheduler.sigmas[i]).astype(np.float32)

        noise_pred = unet.forward(
            (latents_numpy, timestep, text_embeddings_numpy, sigma_numpy)
        )
        noise_pred = torch.from_numpy(noise_pred)
        latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    latents_numpy = latents.detach().numpy()
    image = vae.forward((latents_numpy,))
    image = torch.from_numpy(image)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(f"{args.prompt}.jpg")
