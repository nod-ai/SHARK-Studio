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
import time

GCLOUD_BUCKET = "gs://shark_tank/prashant_nod"
VAE_FP16 = "vae_fp16"
VAE_FP32 = "vae_fp32"
UNET_FP16 = "unet_fp16"
UNET_FP32 = "unet_fp32"
IREE_EXTRA_ARGS = []

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


def get_models():
    global IREE_EXTRA_ARGS
    if args.precision == "fp16":
        IREE_EXTRA_ARGS += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=32",
            "--iree-spirv-unify-aliased-resources=false",
        ]
        if args.import_mlir == True:
            return get_vae16(model_name=VAE_FP16), get_unet16_wrapped(
                model_name=UNET_FP16
            )
        else:
            return get_shark_model(
                GCLOUD_BUCKET,
                VAE_FP16,
                IREE_EXTRA_ARGS,
            ), get_shark_model(
                GCLOUD_BUCKET,
                UNET_FP16,
                IREE_EXTRA_ARGS,
            )

    elif args.precision == "fp32":
        IREE_EXTRA_ARGS += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=16",
        ]
        if args.import_mlir == True:
            return get_vae32(model_name=VAE_FP32), get_unet32_wrapped(
                model_name=UNET_FP32
            )
        else:
            return get_shark_model(
                GCLOUD_BUCKET,
                VAE_FP32,
                IREE_EXTRA_ARGS,
            ), get_shark_model(
                GCLOUD_BUCKET,
                UNET_FP32,
                IREE_EXTRA_ARGS,
            )


if __name__ == "__main__":

    dtype = torch.float32 if args.precision == "fp32" else torch.half
    if len(args.iree_vulkan_target_triple) > 0:
        IREE_EXTRA_ARGS.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )

    prompt = [args.prompt]

    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion

    num_inference_steps = args.steps  # Number of denoising steps

    guidance_scale = args.guidance_scale  # Scale for classifier-free guidance

    generator = torch.manual_seed(
        args.seed
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
        dtype=torch.float32,
    ).to(dtype)

    scheduler.set_timesteps(num_inference_steps)
    scheduler.is_scale_input_called = True

    latents = latents * scheduler.sigmas[0]
    text_embeddings_numpy = text_embeddings.detach().numpy()
    avg_ms = 0

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        step_start = time.time()
        print(f"i = {i} t = {t}", end="")
        timestep = torch.tensor([t]).to(dtype).detach().numpy()
        latents_numpy = latents.detach().numpy()
        sigma_numpy = np.array(scheduler.sigmas[i]).astype(np.float32)

        profile_device = start_profiling(file_path="unet.rdc")
        noise_pred = unet.forward(
            (latents_numpy, timestep, text_embeddings_numpy, sigma_numpy)
        )
        end_profiling(profile_device)
        noise_pred = torch.from_numpy(noise_pred)
        step_time = time.time() - step_start
        avg_ms += step_time
        step_ms = int((step_time) * 1000)
        print(f" ({step_ms}ms)")

        latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
    avg_ms = 1000 * avg_ms / args.steps
    print(f"Average step time: {avg_ms}ms/it")

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    latents_numpy = latents.detach().numpy()
    profile_device = start_profiling(file_path="vae.rdc")
    image = vae.forward((latents_numpy,))
    end_profiling(profile_device)
    image = torch.from_numpy(image)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")

    print("Total image generation runtime (s): {}".format(time.time() - start))

    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(f"{args.prompt}.jpg")
