
# Internal API

# Used for filenames as well as the key for the global cache
def safe_name():
    pass

def local_path():
    pass

def generate_sd_vmfb(
    model: str,
    height: int,
    width: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    batch_size: int = 1,
    base_model_id: str,
    precision: str,
    controlled: bool,
    **kwargs,
):
    pass

def load_sd_vmfb(
    model: str,
    weight_file: str,
    height: int,
    width: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    batch_size: int = 1,
    base_model: str,
    precision: str,
    controlled: bool,
    try_download: bool,
    **kwargs,
):
    # Check if the file is already loaded and cached
    # Check if the file already exists on disk
    # Try to download from the web
    # Generate the vmfb (generate_sd_vmfb)
    # Load the vmfb and weights
    # Return wrapper
    pass

# External API
def generate_images(
    prompt: str,
    negative_prompt: str,
    *,
    height: int = 512,
    width: int = 512,
    steps: int = 20,
    strength: float = 0.8,
    sd_init_image: list = None,
    guidance_scale: float = 7.5,
    seed: list = -1,
    batch_count: int = 1,
    batch_size: int = 1,
    scheduler: str = "EulerDiscrete",
    base_model: str = "sd2",
    custom_weights: str = None,
    custom_vae: str = None,
    precision: str = "fp16",
    device: str = "cpu",
    target_triple: str = None,
    ondemand: bool = False,
    compiled_pipeline: bool = False,
    resample_type: str = "Nearest Neighbor",
    controlnets: dict = {},
    embeddings: dict = {},
    **kwargs,
):
    sd_kwargs = locals()

    # Handle img2img
    if not isinstance(sd_init_image, list):
        sd_init_image = [sd_init_image]
    is_img2img = True if sd_init_image[0] is not None else False

    # Generate seed if < 0
    # TODO

    # Sanity checks
    # Scheduler
    # Base model
    # Custom weights
    # Custom VAE
    # Precision
    # Device
    # Target triple
    # Resample type
    # TODO

    adapters = {}
    is_controlled = False
    control_mode = None
    hints = []
    num_loras = 0
    import_ir = True

    # Populate model map
    if model == "sd1.5":
        submodels = {
            "clip": None,
            "scheduler": None,
            "unet": None,
            "vae_decode": None,
        }
    elif model == "sd2":
        submodels = {
            "clip": None,
            "scheduler": None,
            "unet": None,
            "vae_decode": None,
        }
    elif model == "sdxl":
        submodels = {
            "prompt_encoder": None,
            "scheduled_unet": None,
            "vae_decode": None,
            "pipeline": None,
            "full_pipeline": None,
        }
    elif model == "sd3":
        pass

    # TODO: generate and load submodel vmfbs
    for submodel in submodels:
        submodels[submodel] = load_sd_vmfb(
            submodel,
            custom_weights,
            height,
            width,
            steps,
            strength,
            guidance_scale,
            batch_size,
            model,
            precision,
            not controlnets.keys(),
            True,
        )

    generated_imgs = []
    for current_batch in range(batch_count):

        # TODO: Batch size > 1

        # TODO: random sample (or img2img input)
        sample = None

        # TODO: encode input
        prompt_embeds, negative_prompt_embeds = encode(prompt, negative_prompt)

        start_time = time.time()
        for t in range(steps):
        
            # Prepare latents

            # Scale model input
            latent_model_input = submodels["scheduler"].scale_model_input(
                sample,
                t
            )

            # Run unet
            latents = submodels["unet"](
                latent_model_input,
                t,
                (negative_prompt_embeds, prompt_embeds),
                guidance_scale,
            )

            # Step scheduler
            sample = submodels["scheduler"].step(
                latents,
                t,
                sample
            )

        # VAE decode
        out_img = submodels["vae_decode"](
            sample
        )

        # Processing time
        total_time = time.time() - start_time
        # text_output = f"Total image(s) generation time: {total_time:.4f}sec"
        # print(f"\n[LOG] {text_output}")

        # TODO: Add to output list
        generated_imgs.append(out_img)

        # TODO: Allow the user to halt the process

    return generated_imgs