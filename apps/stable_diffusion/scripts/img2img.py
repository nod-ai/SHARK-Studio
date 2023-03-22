import sys
import torch
import time
from PIL import Image
import transformers
from apps.stable_diffusion.src import (
    args,
    Image2ImagePipeline,
    StencilPipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
    clear_all,
    save_output_img,
)
from apps.stable_diffusion.src.utils import get_generation_text_info


# set initial values of iree_vulkan_target_triple, use_tuned and import_mlir.
init_iree_vulkan_target_triple = args.iree_vulkan_target_triple
init_use_tuned = args.use_tuned
init_import_mlir = args.import_mlir


# For stencil, the input image can be of any size but we need to ensure that
# it conforms with our model contraints :-
#   Both width and height should be > 384 and multiple of 8.
# This utility function performs the transformation on the input image while
# also maintaining the aspect ratio before sending it to the stencil pipeline.
def resize_stencil(image: Image.Image):
    width, height = image.size
    aspect_ratio = width / height
    min_size = min(width, height)
    if min_size < 384:
        n_size = 384
        if width == min_size:
            width = n_size
            height = n_size / aspect_ratio
        else:
            height = n_size
            width = n_size * aspect_ratio
    width = int(width)
    height = int(height)
    n_width = width // 8
    n_height = height // 8
    n_width *= 8
    n_height *= 8
    new_image = image.resize((n_width, n_height))
    return new_image, n_width, n_height


# Exposed to UI.
def img2img_inf(
    prompt: str,
    negative_prompt: str,
    init_image,
    height: int,
    width: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    seed: int,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    custom_model: str,
    hf_model_id: str,
    precision: str,
    device: str,
    max_length: int,
    use_stencil: str,
    save_metadata_to_json: bool,
    save_metadata_to_png: bool,
    lora_weights: str,
    lora_hf_id: str,
):
    from apps.stable_diffusion.web.ui.utils import (
        get_custom_model_pathfile,
        get_custom_vae_or_lora_weights,
        Config,
    )
    import apps.stable_diffusion.web.utils.global_obj as global_obj
    from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
        SD_STATE_CANCEL,
    )

    args.prompts = [prompt]
    args.negative_prompts = [negative_prompt]
    args.guidance_scale = guidance_scale
    args.seed = seed
    args.steps = steps
    args.strength = strength
    args.scheduler = scheduler
    args.img_path = "not none"

    if init_image is None:
        return None, "An Initial Image is required"
    image = init_image.convert("RGB")

    # set ckpt_loc and hf_model_id.
    args.ckpt_loc = ""
    args.hf_model_id = ""
    if custom_model == "None":
        if not hf_model_id:
            return (
                None,
                "Please provide either custom model or huggingface model ID, both must not be empty",
            )
        args.hf_model_id = hf_model_id
    elif ".ckpt" in custom_model or ".safetensors" in custom_model:
        args.ckpt_loc = get_custom_model_pathfile(custom_model)
    else:
        args.hf_model_id = custom_model

    args.use_lora = get_custom_vae_or_lora_weights(
        lora_weights, lora_hf_id, "lora"
    )

    args.save_metadata_to_json = save_metadata_to_json
    args.write_metadata_to_png = save_metadata_to_png

    use_stencil = None if use_stencil == "None" else use_stencil
    args.use_stencil = use_stencil
    if use_stencil is not None:
        args.scheduler = "DDIM"
        args.hf_model_id = "runwayml/stable-diffusion-v1-5"
        image, width, height = resize_stencil(image)
    elif args.scheduler != "PNDM":
        if "Shark" in args.scheduler:
            print(
                f"SharkEulerDiscrete scheduler not supported. Switching to PNDM scheduler"
            )
            args.scheduler = "PNDM"
        else:
            sys.exit(
                "Img2Img works best with PNDM scheduler. Other schedulers are not supported yet."
            )
    cpu_scheduling = not args.scheduler.startswith("Shark")
    args.precision = precision
    dtype = torch.float32 if precision == "fp32" else torch.half
    new_config_obj = Config(
        "img2img",
        args.hf_model_id,
        args.ckpt_loc,
        precision,
        batch_size,
        max_length,
        height,
        width,
        device,
        use_lora=args.use_lora,
        use_stencil=use_stencil,
    )
    if (
        not global_obj.get_sd_obj()
        or global_obj.get_cfg_obj() != new_config_obj
    ):
        global_obj.clear_cache()
        global_obj.set_cfg_obj(new_config_obj)
        args.batch_count = batch_count
        args.batch_size = batch_size
        args.max_length = max_length
        args.height = height
        args.width = width
        args.device = device.split("=>", 1)[1].strip()
        args.iree_vulkan_target_triple = init_iree_vulkan_target_triple
        args.use_tuned = init_use_tuned
        args.import_mlir = init_import_mlir
        set_init_device_flags()
        model_id = (
            args.hf_model_id
            if args.hf_model_id
            else "stabilityai/stable-diffusion-2-1-base"
        )
        global_obj.set_schedulers(get_schedulers(model_id))
        scheduler_obj = global_obj.get_scheduler(scheduler)

        if use_stencil is not None:
            args.use_tuned = False
            global_obj.set_sd_obj(
                StencilPipeline.from_pretrained(
                    scheduler_obj,
                    args.import_mlir,
                    args.hf_model_id,
                    args.ckpt_loc,
                    args.custom_vae,
                    args.precision,
                    args.max_length,
                    args.batch_size,
                    args.height,
                    args.width,
                    args.use_base_vae,
                    args.use_tuned,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_stencil=use_stencil,
                    debug=args.import_debug if args.import_mlir else False,
                    use_lora=args.use_lora,
                )
            )
        else:
            global_obj.set_sd_obj(
                Image2ImagePipeline.from_pretrained(
                    scheduler_obj,
                    args.import_mlir,
                    args.hf_model_id,
                    args.ckpt_loc,
                    args.custom_vae,
                    args.precision,
                    args.max_length,
                    args.batch_size,
                    args.height,
                    args.width,
                    args.use_base_vae,
                    args.use_tuned,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    debug=args.import_debug if args.import_mlir else False,
                    use_lora=args.use_lora,
                )
            )

    global_obj.set_sd_scheduler(scheduler)

    start_time = time.time()
    global_obj.get_sd_obj().log = ""
    generated_imgs = []
    seeds = []
    img_seed = utils.sanitize_seed(seed)
    extra_info = {"STRENGTH": strength}
    text_output = ""
    for current_batch in range(batch_count):
        if current_batch > 0:
            img_seed = utils.sanitize_seed(-1)
        out_imgs = global_obj.get_sd_obj().generate_images(
            prompt,
            negative_prompt,
            image,
            batch_size,
            height,
            width,
            steps,
            strength,
            guidance_scale,
            img_seed,
            args.max_length,
            dtype,
            args.use_base_vae,
            cpu_scheduling,
            use_stencil=use_stencil,
        )
        seeds.append(img_seed)
        total_time = time.time() - start_time
        text_output = get_generation_text_info(seeds, device)
        text_output += "\n" + global_obj.get_sd_obj().log
        text_output += f"\nTotal image(s) generation time: {total_time:.4f}sec"

        if global_obj.get_sd_status() == SD_STATE_CANCEL:
            break
        else:
            save_output_img(out_imgs[0], img_seed, extra_info)
            generated_imgs.extend(out_imgs)
            yield generated_imgs, text_output

    return generated_imgs, text_output


if __name__ == "__main__":
    if args.clear_all:
        clear_all()

    if args.img_path is None:
        print("Flag --img_path is required.")
        exit()

    image = Image.open(args.img_path).convert("RGB")
    # When the models get uploaded, it should be default to False.
    args.import_mlir = True

    use_stencil = args.use_stencil
    if use_stencil:
        args.scheduler = "DDIM"
        args.hf_model_id = "runwayml/stable-diffusion-v1-5"
        image, args.width, args.height = resize_stencil(image)
    elif args.scheduler != "PNDM":
        if "Shark" in args.scheduler:
            print(
                f"SharkEulerDiscrete scheduler not supported. Switching to PNDM scheduler"
            )
            args.scheduler = "PNDM"
        else:
            sys.exit(
                "Img2Img works best with PNDM scheduler. Other schedulers are not supported yet."
            )
    cpu_scheduling = not args.scheduler.startswith("Shark")
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)
    scheduler_obj = schedulers[args.scheduler]
    seed = utils.sanitize_seed(args.seed)
    # Adjust for height and width based on model

    if use_stencil:
        img2img_obj = StencilPipeline.from_pretrained(
            scheduler_obj,
            args.import_mlir,
            args.hf_model_id,
            args.ckpt_loc,
            args.custom_vae,
            args.precision,
            args.max_length,
            args.batch_size,
            args.height,
            args.width,
            args.use_base_vae,
            args.use_tuned,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            use_stencil=use_stencil,
            debug=args.import_debug if args.import_mlir else False,
            use_lora=args.use_lora,
        )
    else:
        img2img_obj = Image2ImagePipeline.from_pretrained(
            scheduler_obj,
            args.import_mlir,
            args.hf_model_id,
            args.ckpt_loc,
            args.custom_vae,
            args.precision,
            args.max_length,
            args.batch_size,
            args.height,
            args.width,
            args.use_base_vae,
            args.use_tuned,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            debug=args.import_debug if args.import_mlir else False,
            use_lora=args.use_lora,
        )

    start_time = time.time()
    generated_imgs = img2img_obj.generate_images(
        args.prompts,
        args.negative_prompts,
        image,
        args.batch_size,
        args.height,
        args.width,
        args.steps,
        args.strength,
        args.guidance_scale,
        seed,
        args.max_length,
        dtype,
        args.use_base_vae,
        cpu_scheduling,
        use_stencil=use_stencil,
    )
    total_time = time.time() - start_time
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
    text_output += f"\nscheduler={args.scheduler}, device={args.device}"
    text_output += f"\nsteps={args.steps}, strength={args.strength}, guidance_scale={args.guidance_scale}, seed={seed}, size={args.height}x{args.width}"
    text_output += (
        f", batch size={args.batch_size}, max_length={args.max_length}"
    )
    text_output += img2img_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    extra_info = {"STRENGTH": args.strength}
    save_output_img(generated_imgs[0], seed, extra_info)
    print(text_output)
