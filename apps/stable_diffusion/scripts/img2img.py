import sys
import torch
import time
from PIL import Image
from dataclasses import dataclass
from apps.stable_diffusion.src import (
    args,
    Image2ImagePipeline,
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


img2img_obj = None
config_obj = None
schedulers = None


# Exposed to UI.
def img2img_inf(
    prompt: str,
    negative_prompt: str,
    init_image: Image,
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
    save_metadata_to_json: bool,
    save_metadata_to_png: bool,
):
    global img2img_obj
    global config_obj
    global schedulers

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
        args.ckpt_loc = custom_model
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
    if not img2img_obj or config_obj != new_config_obj:
        config_obj = new_config_obj
        args.precision = precision
        args.batch_size = batch_size
        args.max_length = max_length
        args.height = height
        args.width = width
        args.device = device.split("=>", 1)[1].strip()
        args.iree_vulkan_target_triple = ""
        args.use_tuned = True
        args.import_mlir = True
        set_init_device_flags()
        model_id = (
            args.hf_model_id
            if args.hf_model_id
            else "stabilityai/stable-diffusion-2-1-base"
        )
        schedulers = get_schedulers(model_id)
        scheduler_obj = schedulers[scheduler]
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
        )

    img2img_obj.scheduler = schedulers[scheduler]

    start_time = time.time()
    img2img_obj.log = ""
    generated_imgs = []
    seeds = []
    img_seed = utils.sanitize_seed(seed)
    extra_info = {"STRENGTH": strength}
    for current_batch in range(batch_count):
        if current_batch > 0:
            img_seed = utils.sanitize_seed(-1)
        out_imgs = img2img_obj.generate_images(
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
        )
        save_output_img(out_imgs[0], img_seed, extra_info)
        generated_imgs.extend(out_imgs)
        seeds.append(img_seed)
        img2img_obj.log += "\n"

    total_time = time.time() - start_time
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
    text_output += f"\nscheduler={args.scheduler}, device={device}"
    text_output += f"\nsteps={steps}, strength={args.strength}, guidance_scale={guidance_scale}, seed={seeds}"
    text_output += f"\nsize={height}x{width}, batch_count={batch_count}, batch_size={batch_size}, max_length={args.max_length}"
    text_output += img2img_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    return generated_imgs, text_output


if __name__ == "__main__":
    if args.clear_all:
        clear_all()

    if args.img_path is None:
        print("Flag --img_path is required.")
        exit()

    # When the models get uploaded, it should be default to False.
    args.import_mlir = True

    dtype = torch.float32 if args.precision == "fp32" else torch.half
    cpu_scheduling = not args.scheduler.startswith("Shark")
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)
    if args.scheduler != "PNDM":
        if "Shark" in args.scheduler:
            print(
                f"SharkEulerDiscrete scheduler not supported. Switching to PNDM scheduler"
            )
            args.scheduler = "PNDM"
        else:
            sys.exit(
                "Img2Img works best with PNDM scheduler. Other schedulers are not supported yet."
            )

    scheduler_obj = schedulers[args.scheduler]
    image = Image.open(args.img_path).convert("RGB")
    seed = utils.sanitize_seed(args.seed)

    # Adjust for height and width based on model

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
