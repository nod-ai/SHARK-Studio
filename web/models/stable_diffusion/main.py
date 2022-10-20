from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
import numpy as np
from models.stable_diffusion.model_wrappers import (
    get_vae32,
    get_vae16,
    get_unet16_wrapped,
    get_unet32_wrapped,
)
from models.stable_diffusion.utils import get_shark_model
import time
import os

GCLOUD_BUCKET = "gs://shark_tank/prashant_nod"
VAE_FP16 = "vae_fp16"
VAE_FP32 = "vae_fp32"
UNET_FP16 = "unet_fp16"
UNET_FP32 = "unet_fp32"

args = None
DEBUG = False


class Arguments:
    def __init__(
        self,
        prompt="a boy riding a bicycle",
        steps=10,
        precision="fp32",
        device="cpu",
        import_mlir=False,
        max_length=77,
    ):
        self.prompt = prompt
        self.steps = steps
        self.precision = precision
        self.device = device
        self.import_mlir = import_mlir
        self.max_length = max_length


def get_models():
    if args.precision == "fp16":
        if args.import_mlir == True:
            return get_unet16_wrapped(args), get_vae16(args)
        return get_shark_model(args, GCLOUD_BUCKET, VAE_FP16), get_shark_model(
            args, GCLOUD_BUCKET, UNET_FP16
        )

    elif args.precision == "fp32":
        if args.import_mlir == True:
            return (get_vae32(args), get_unet32_wrapped(args))
        return get_shark_model(args, GCLOUD_BUCKET, VAE_FP32), get_shark_model(
            args,
            GCLOUD_BUCKET,
            UNET_FP32,
            [
                "--iree-flow-enable-conv-nchw-to-nhwc-transform",
                "--iree-flow-enable-padding-linalg-ops",
                "--iree-flow-linalg-ops-padding-size=16",
            ],
        )
    return None, None


def stable_diff_inf(prompt: str, steps, precision, device: str):

    global args
    global DEBUG

    output_loc = f"stored_results/stable_diffusion/{prompt}_{int(steps)}_{precision}_{device}.jpg"
    DEBUG = False
    log_write = open(r"logs/stable_diffusion_log.txt", "w")
    if log_write:
        DEBUG = True

    args = Arguments(prompt, steps, precision, device)
    dtype = torch.float32 if args.precision == "fp32" else torch.half

    prompt = [args.prompt]

    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion

    num_inference_steps = int(args.steps)  # Number of denoising steps

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

    start = time.time()

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

    avg_ms = 0
    for i, t in tqdm(enumerate(scheduler.timesteps)):

        if DEBUG:
            log_write.write(f"\ni = {i} t = {t} ")
        step_start = time.time()
        timestep = torch.tensor([t]).to(dtype).detach().numpy()
        latents_numpy = latents.detach().numpy()
        sigma_numpy = np.array(scheduler.sigmas[i]).astype(np.float32)

        noise_pred = unet.forward(
            (latents_numpy, timestep, text_embeddings_numpy, sigma_numpy)
        )
        noise_pred = torch.from_numpy(noise_pred)
        step_time = time.time() - step_start
        avg_ms += step_time
        step_ms = int((step_time) * 1000)
        if DEBUG:
            log_write.write(f"time/itr={step_ms}ms")
        latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

    avg_ms = 1000 * avg_ms / args.steps
    if DEBUG:
        log_write.write(f"\nAverage step time: {avg_ms}ms/it")

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    latents_numpy = latents.detach().numpy()
    image = vae.forward((latents_numpy,))
    image = torch.from_numpy(image)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    output = pil_images[0]
    # save the output image with the prompt name.
    output.save(os.path.join(output_loc))
    log_write.close()

    std_output = ""
    with open(r"logs/stable_diffusion_log.txt", "r") as log_read:
        std_output = log_read.read()
    return pil_images[0], std_output
