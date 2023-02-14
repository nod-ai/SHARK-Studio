import os

if "AMD_ENABLE_LLPC" not in os.environ:
    os.environ["AMD_ENABLE_LLPC"] = "0"

import sys
import json
import torch
import re
import time
from pathlib import Path
from PIL import Image, PngImagePlugin
from datetime import datetime as dt
from dataclasses import dataclass
from csv import DictWriter
from apps.stable_diffusion.src import (
    args,
    Image2ImagePipeline,
    get_schedulers,
    set_init_device_flags,
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


# This has to come before importing cache objects
if args.clear_all:
    print("CLEARING ALL, EXPECT SEVERAL MINUTES TO RECOMPILE")
    from glob import glob
    import shutil

    vmfbs = glob(os.path.join(os.getcwd(), "*.vmfb"))
    for vmfb in vmfbs:
        if os.path.exists(vmfb):
            os.remove(vmfb)
    # Temporary workaround of deleting yaml files to incorporate diffusers' pipeline.
    # TODO: Remove this once we have better weight updation logic.
    inference_yaml = ["v2-inference-v.yaml", "v1-inference.yaml"]
    for yaml in inference_yaml:
        if os.path.exists(yaml):
            os.remove(yaml)
    home = os.path.expanduser("~")
    if os.name == "nt":  # Windows
        appdata = os.getenv("LOCALAPPDATA")
        shutil.rmtree(os.path.join(appdata, "AMD/VkCache"), ignore_errors=True)
        shutil.rmtree(os.path.join(home, "shark_tank"), ignore_errors=True)
    elif os.name == "unix":
        shutil.rmtree(os.path.join(home, ".cache/AMD/VkCache"))
        shutil.rmtree(os.path.join(home, ".local/shark_tank"))


# save output images and the inputs correspoding to it.
def save_output_img(output_img):
    output_path = args.output_dir if args.output_dir else Path.cwd()
    generated_imgs_path = Path(output_path, "generated_imgs")
    generated_imgs_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(generated_imgs_path, "imgs_details.csv")

    prompt_slice = re.sub("[^a-zA-Z0-9]", "_", args.prompts[0][:15])
    out_img_name = (
        f"{prompt_slice}_{args.seed}_{dt.now().strftime('%y%m%d_%H%M%S')}"
    )

    if args.output_img_format == "jpg":
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.jpg")
        output_img.save(out_img_path, quality=95, subsampling=0)
    else:
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.png")
        pngInfo = PngImagePlugin.PngInfo()

        if args.write_metadata_to_png:
            pngInfo.add_text(
                "parameters",
                f"{args.prompts[0]}\nNegative prompt: {args.negative_prompts[0]}\nSteps:{args.steps}, Sampler: {args.scheduler}, CFG scale: {args.guidance_scale}, Seed: {args.seed}, Size: {args.width}x{args.height}, Model: {args.hf_model_id}",
            )

        output_img.save(out_img_path, "PNG", pnginfo=pngInfo)

        if args.output_img_format not in ["png", "jpg"]:
            print(
                f"[ERROR] Format {args.output_img_format} is not supported yet."
                "Image saved as png instead. Supported formats: png / jpg"
            )

    new_entry = {
        "VARIANT": args.hf_model_id,
        "SCHEDULER": args.scheduler,
        "PROMPT": args.prompts[0],
        "NEG_PROMPT": args.negative_prompts[0],
        "IMG_INPUT": args.img_path,
        "SEED": args.seed,
        "CFG_SCALE": args.guidance_scale,
        "PRECISION": args.precision,
        "STEPS": args.steps,
        "HEIGHT": args.height,
        "WIDTH": args.width,
        "MAX_LENGTH": args.max_length,
        "OUTPUT": out_img_path,
    }

    with open(csv_path, "a") as csv_obj:
        dictwriter_obj = DictWriter(csv_obj, fieldnames=list(new_entry.keys()))
        dictwriter_obj.writerow(new_entry)
        csv_obj.close()

    if args.save_metadata_to_json:
        del new_entry["OUTPUT"]
        json_path = Path(generated_imgs_path, f"{out_img_name}.json")
        with open(json_path, "w") as f:
            json.dump(new_entry, f, indent=4)


img2img_obj = None
config_obj = None
schedulers = None


# Exposed to UI.
def img2img_inf(
    prompt: str,
    negative_prompt: str,
    init_image: str,
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
    global img2img_obj
    global config_obj
    global schedulers

    args.prompts = [prompt]
    args.negative_prompts = [negative_prompt]
    args.guidance_scale = guidance_scale
    args.seed = seed
    args.steps = steps
    args.scheduler = scheduler
    args.img_path = init_image
    image = Image.open(args.img_path)

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

    if image is None:
        return None, "An Initial Image is required"

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
    if config_obj != new_config_obj:
        config_obj = new_config_obj
        args.precision = precision
        args.batch_size = batch_size
        args.max_length = max_length
        args.height = height
        args.width = width
        args.device = device.split("=>", 1)[1].strip()
        args.use_tuned = True
        args.import_mlir = True
        set_init_device_flags()
        model_id = (
            args.hf_model_id
            if args.hf_model_id
            else "runwayml/stable-diffusion-inpainting"
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
        )

    if not img2img_obj:
        sys.exit("text to image pipeline must not return a null value")

    img2img_obj.scheduler = schedulers[scheduler]

    start_time = time.time()
    img2img_obj.log = ""
    generated_imgs = img2img_obj.generate_images(
        prompt,
        negative_prompt,
        image,
        batch_size,
        height,
        width,
        steps,
        guidance_scale,
        seed,
        args.max_length,
        dtype,
        args.use_base_vae,
        cpu_scheduling,
    )
    total_time = time.time() - start_time
    save_output_img(generated_imgs[0])
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
    text_output += f"\nscheduler={args.scheduler}, device={device}"
    text_output += f"\nsteps={args.steps}, guidance_scale={args.guidance_scale}, seed={args.seed}, size={args.height}x{args.width}"
    text_output += (
        f", batch size={args.batch_size}, max_length={args.max_length}"
    )
    text_output += img2img_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    return generated_imgs, text_output


if __name__ == "__main__":
    if args.img_path is None:
        print("Flag --img_path is required.")
        exit()

    # When the models get uploaded, it should be default to False.
    args.import_mlir = True

    dtype = torch.float32 if args.precision == "fp32" else torch.half
    cpu_scheduling = not args.scheduler.startswith("Shark")
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)
    scheduler_obj = schedulers[args.scheduler]
    image = Image.open(args.img_path)

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
        args.guidance_scale,
        args.seed,
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
    text_output += f"\nsteps={args.steps}, guidance_scale={args.guidance_scale}, seed={args.seed}, size={args.height}x{args.width}"
    text_output += (
        f", batch size={args.batch_size}, max_length={args.max_length}"
    )
    text_output += img2img_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    save_output_img(generated_imgs[0])
    print(text_output)
