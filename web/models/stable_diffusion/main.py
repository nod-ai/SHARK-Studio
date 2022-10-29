from transformers import CLIPTextModel, CLIPTokenizer
import torch
from PIL import Image
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
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

IREE_EXTRA_ARGS = []
args = None
DEBUG = False


class Arguments:
    def __init__(
        self,
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
        load_vmfb: bool,
        save_vmfb: bool,
        iree_vulkan_target_triple: str,
        import_mlir: bool = False,
        max_length: int = 77,
    ):
        self.prompt = prompt
        self.scheduler = scheduler
        self.iteration_count = iteration_count
        self.batch_size = batch_size
        self.steps = steps
        self.guidance = guidance
        self.height = height
        self.width = width
        self.seed = seed
        self.precision = precision
        self.device = device
        self.load_vmfb = load_vmfb
        self.save_vmfb = save_vmfb
        self.iree_vulkan_target_triple = iree_vulkan_target_triple
        self.import_mlir = import_mlir
        self.max_length = max_length


def get_models():

    global IREE_EXTRA_ARGS
    global args

    if args.precision == "fp16":
        IREE_EXTRA_ARGS += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=32",
        ]
        if args.import_mlir == True:
            return get_vae16(args, model_name=VAE_FP16), get_unet16_wrapped(
                args, model_name=UNET_FP16
            )
        return get_shark_model(
            args, GCLOUD_BUCKET, VAE_FP16, IREE_EXTRA_ARGS
        ), get_shark_model(args, GCLOUD_BUCKET, UNET_FP16, IREE_EXTRA_ARGS)

    elif args.precision == "fp32":
        IREE_EXTRA_ARGS += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=16",
        ]
        if args.import_mlir == True:
            return (
                get_vae32(args, model_name=VAE_FP32),
                get_unet32_wrapped(args, model_name=UNET_FP32),
            )
        return get_shark_model(
            args, GCLOUD_BUCKET, VAE_FP32, IREE_EXTRA_ARGS
        ), get_shark_model(args, GCLOUD_BUCKET, UNET_FP32, IREE_EXTRA_ARGS)
    return None, None


def stable_diff_inf(
    prompt: str,
    scheduler: str,
    iteration_count: int,
    batch_size: int,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    seed: str,
    precision: str,
    device: str,
    load_vmfb: bool,
    save_vmfb: bool,
    iree_vulkan_target_triple: str,
):

    global IREE_EXTRA_ARGS
    global args
    global DEBUG

    output_loc = f"stored_results/stable_diffusion/{prompt}_{int(steps)}_{precision}_{device}.jpg"
    DEBUG = False
    log_write = open(r"logs/stable_diffusion_log.txt", "w")
    if log_write:
        DEBUG = True

    # set seed value
    if seed == "":
        seed = int(torch.randint(low=25, high=100, size=()))
    else:
        try:
            seed = int(seed)
        except ValueError:
            seed = hash(seed)

    # set scheduler value
    if scheduler == "PNDM":
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
    elif scheduler == "LMS":
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
    elif scheduler == "DDIM":
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    else:
        raise Exception(
            f"Does not support scheduler with name {args.scheduler}."
        )

    args = Arguments(
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
        load_vmfb,
        save_vmfb,
        iree_vulkan_target_triple,
    )
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    if len(args.iree_vulkan_target_triple) > 0:
        IREE_EXTRA_ARGS.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    num_inference_steps = int(args.steps)  # Number of denoising steps
    generator = torch.manual_seed(
        args.seed
    )  # Seed generator to create the inital latent noise

    vae, unet = get_models()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    text_input = tokenizer(
        [args.prompt] * batch_size,
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
        (batch_size, 4, args.height // 8, args.width // 8),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)

    scheduler.set_timesteps(num_inference_steps)
    scheduler.is_scale_input_called = True

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
            log_write.write(f"time={step_ms}ms")
        latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    latents_numpy = latents.detach().numpy()
    image = vae.forward((latents_numpy,))
    image = torch.from_numpy(image)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    avg_ms = 1000 * avg_ms / args.steps
    if DEBUG:
        log_write.write(f"\nAverage step time: {avg_ms}ms/it")

    print("total images:", len(pil_images))
    output = pil_images[0]
    # save the output image with the prompt name.
    output.save(os.path.join(output_loc))
    log_write.close()

    std_output = ""
    with open(r"logs/stable_diffusion_log.txt", "r") as log_read:
        std_output = log_read.read()
    return output, std_output
