import torch
import time
from PIL import Image
from dataclasses import dataclass
from apps.stable_diffusion.src import (
    args,
    OutpaintPipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
    clear_all,
    save_output_img,
)


@dataclass
class Config:
    model_id: str
    ckpt_loc: str
    precision: str
    batch_size: int
    max_length: int
    height: int
    width: int
    device: str


outpaint_obj = None
config_obj = None
schedulers = None

# set initial values of iree_vulkan_target_triple, use_tuned and import_mlir.
init_iree_vulkan_target_triple = args.iree_vulkan_target_triple
init_use_tuned = args.use_tuned
init_import_mlir = args.import_mlir


# Exposed to UI.
def outpaint_inf(
    prompt: str,
    negative_prompt: str,
    init_image,
    pixels: int,
    mask_blur: int,
    directions: list,
    noise_q: float,
    color_variation: float,
    height: int,
    width: int,
    steps: int,
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
    save_metadata_to_json: bool,
    save_metadata_to_png: bool,
):
    from apps.stable_diffusion.web.ui.utils import get_custom_model_pathfile

    global outpaint_obj
    global config_obj
    global schedulers

    args.prompts = [prompt]
    args.negative_prompts = [negative_prompt]
    args.guidance_scale = guidance_scale
    args.steps = steps
    args.scheduler = scheduler
    args.img_path = "not none"

    # set ckpt_loc and hf_model_id.
    types = (
        ".ckpt",
        ".safetensors",
    )  # the tuple of file types
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

    args.save_metadata_to_json = save_metadata_to_json
    args.write_metadata_to_png = save_metadata_to_png

    dtype = torch.float32 if precision == "fp32" else torch.half
    cpu_scheduling = not scheduler.startswith("Shark")
    new_config_obj = Config(
        args.hf_model_id,
        args.ckpt_loc,
        precision,
        batch_size,
        max_length,
        height,
        width,
        device,
    )
    if not outpaint_obj or config_obj != new_config_obj:
        config_obj = new_config_obj
        args.precision = precision
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
            else "stabilityai/stable-diffusion-2-inpainting"
        )
        schedulers = get_schedulers(model_id)
        scheduler_obj = schedulers[scheduler]
        outpaint_obj = OutpaintPipeline.from_pretrained(
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
        )

    outpaint_obj.scheduler = schedulers[scheduler]

    start_time = time.time()
    outpaint_obj.log = ""
    generated_imgs = []
    seeds = []
    img_seed = utils.sanitize_seed(seed)

    left = True if "left" in directions else False
    right = True if "right" in directions else False
    top = True if "up" in directions else False
    bottom = True if "down" in directions else False

    for i in range(batch_count):
        if i > 0:
            img_seed = utils.sanitize_seed(-1)
        out_imgs = outpaint_obj.generate_images(
            prompt,
            negative_prompt,
            init_image,
            pixels,
            mask_blur,
            left,
            right,
            top,
            bottom,
            noise_q,
            color_variation,
            batch_size,
            height,
            width,
            steps,
            guidance_scale,
            img_seed,
            args.max_length,
            dtype,
            args.use_base_vae,
            cpu_scheduling,
        )
        save_output_img(out_imgs[0], img_seed)
        generated_imgs.extend(out_imgs)
        seeds.append(img_seed)
        outpaint_obj.log += "\n"
        yield generated_imgs, generated_imgs[0], outpaint_obj.log

    total_time = time.time() - start_time
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
    text_output += f"\nscheduler={args.scheduler}, device={device}"
    text_output += f"\nsteps={args.steps}, guidance_scale={args.guidance_scale}, seed={seeds}"
    text_output += f"\nsize={args.height}x{args.width}, batch-count={batch_count}, batch-size={args.batch_size}, max_length={args.max_length}"
    text_output += outpaint_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    yield generated_imgs, text_output


if __name__ == "__main__":
    if args.clear_all:
        clear_all()

    if args.img_path is None:
        print("Flag --img_path is required.")
        exit()

    dtype = torch.float32 if args.precision == "fp32" else torch.half
    cpu_scheduling = not args.scheduler.startswith("Shark")
    set_init_device_flags()
    model_id = (
        args.hf_model_id
        if "inpaint" in args.hf_model_id
        else "stabilityai/stable-diffusion-2-inpainting"
    )
    schedulers = get_schedulers(model_id)
    scheduler_obj = schedulers[args.scheduler]
    seed = args.seed
    image = Image.open(args.img_path)

    outpaint_obj = OutpaintPipeline.from_pretrained(
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
    )

    for current_batch in range(args.batch_count):
        if current_batch > 0:
            seed = -1
        seed = utils.sanitize_seed(seed)

        start_time = time.time()
        generated_imgs = outpaint_obj.generate_images(
            args.prompts,
            args.negative_prompts,
            image,
            args.pixels,
            args.mask_blur,
            args.left,
            args.right,
            args.top,
            args.bottom,
            args.noise_q,
            args.color_variation,
            args.batch_size,
            args.height,
            args.width,
            args.steps,
            args.guidance_scale,
            seed,
            args.max_length,
            dtype,
            args.use_base_vae,
            cpu_scheduling,
        )
        total_time = time.time() - start_time
        text_output = f"prompt={args.prompts}"
        text_output += f"\nnegative prompt={args.negative_prompts}"
        text_output += (
            f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
        )
        text_output += f"\nscheduler={args.scheduler}, device={args.device}"
        text_output += f"\nsteps={args.steps}, guidance_scale={args.guidance_scale}, seed={seed}, size={args.height}x{args.width}"
        text_output += (
            f", batch size={args.batch_size}, max_length={args.max_length}"
        )
        text_output += outpaint_obj.log
        text_output += f"\nTotal image generation time: {total_time:.4f}sec"

        # save this information as metadata of output generated image.
        directions = []
        if args.left:
            directions.append("left")
        if args.right:
            directions.append("right")
        if args.top:
            directions.append("up")
        if args.bottom:
            directions.append("down")
        extra_info = {
            "PIXELS": args.pixels,
            "MASK_BLUR": args.mask_blur,
            "DIRECTIONS": directions,
            "NOISE_Q": args.noise_q,
            "COLOR_VARIATION": args.color_variation,
        }
        save_output_img(generated_imgs[0], seed, extra_info)
        print(text_output)
