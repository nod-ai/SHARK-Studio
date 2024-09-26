# Internal API
pipelines = {
    "sd1.5": ("", None),
    "sd2": ("", None),
    "sdxl": ("", None),
    "sd3": ("", None),
}


# Used for filenames as well as the key for the global cache
def safe_name(
    model_name: str,
    height: int,
    width: int,
    batch_size: int,
):
    pass


def local_path():
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
        sd_init_image = [sd_init_image] * batch_count
    is_img2img = True if sd_init_image[0] is not None else False

    # Generate seed if < 0
    # TODO

    # Cache dir
    # TODO
    pipeline_dir = None

    # Sanity checks
    assert scheduler in ["EulerDiscrete"]
    assert base_model in ["sd1.5", "sd2", "sdxl", "sd3"]
    assert precision in ["fp16", "fp32"]
    assert device in [
        "cpu",
        "vulkan",
        "rocm",
        "hip",
        "cuda",
    ]  # and (IREE check if the device exists)
    assert resample_type in ["Nearest Neighbor"]

    # Custom weights
    # TODO
    # Custom VAE
    # TODO
    # Target triple
    # TODO

    # (Re)initialize pipeline
    pipeline_args = {
        "height": height,
        "width": width,
        "batch_size": batch_size,
        "precision": precision,
        "device": device,
        "target_triple": target_triple,
    }
    (existing_args, pipeline) = pipelines[base_model]
    if not existing_args or not pipeline or not pipeline_args == existing_args:
        # TODO: Initialize new pipeline
        if base_model in ["sd1.5", "sd2"]:
            new_pipeline = SharkSDPipeline(
                hf_model_name=("stabilityai/stable-diffusion-2-1" if base_model == "sd2" else "stabilityai/stable-diffusion-1-5"),
                scheduler_id=scheduler,
                height=height,
                width=width,
                precision=precision,
                max_length=64,
                batch_size=batch_size,
                num_inference_steps=steps,
                device=device,  # TODO: Get the IREE device ID?
                iree_target_triple=target_triple,
                ireec_flags={},
                attn_spec=None,  # TODO: Find a better way to figure this out than hardcoding
                decomp_attn=True,  # TODO: Ditto
                pipeline_dir=pipeline_dir,
                external_weights_dir=weights,  # TODO: Are both necessary still?
                external_weights=weights,
                custom_vae=custom_vae,
            )
        elif base_model == "sdxl":
            pass
        elif base_model == "sd3":
            pass
        existing_args = pipeline_args
        pipeline = new_pipeline
        pipelines[base_model] = (existing_args, pipeline)

    generated_images = []
    for current_batch in range(batch_count):

        start_time = time.time()
        for t in range(steps):

            out_images = pipeline.generate_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=sd_init_image[current_batch],
                strength=strength,
                guidance_scale=guidance_scale,
                seed=seed,
                ondemand=ondemand,
                resample_type=resample_type,
                control_mode=control_mode,
                hints=hints,
            )

        # Processing time
        total_time = time.time() - start_time
        # text_output = f"Total image(s) generation time: {total_time:.4f}sec"
        # print(f"\n[LOG] {text_output}")

        # TODO: Add to output list
        if not isinstance(out_images, list):
            out_images = [out_images]
        generated_images.extend(out_images)

        # TODO: Allow the user to halt the process

    return generated_images
