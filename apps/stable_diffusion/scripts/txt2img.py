import os

if "AMD_ENABLE_LLPC" not in os.environ:
    os.environ["AMD_ENABLE_LLPC"] = "1"

import sys
import json
import torch
import re
import time
from pathlib import Path
from PIL import PngImagePlugin
from datetime import datetime as dt
from dataclasses import dataclass
from csv import DictWriter
from apps.stable_diffusion.src import (
    args,
    Text2ImagePipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
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


# save output images and the inputs corresponding to it.
def save_output_img(output_img, img_seed):
    output_path = args.output_dir if args.output_dir else Path.cwd()
    generated_imgs_path = Path(output_path, "generated_imgs")
    generated_imgs_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(generated_imgs_path, "imgs_details.csv")

    prompt_slice = re.sub("[^a-zA-Z0-9]", "_", args.prompts[0][:15])
    out_img_name = (
        f"{prompt_slice}_{img_seed}_{dt.now().strftime('%y%m%d_%H%M%S')}"
    )

    img_model = args.hf_model_id
    if args.ckpt_loc:
        img_model = os.path.basename(args.ckpt_loc)

    if args.output_img_format == "jpg":
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.jpg")
        output_img.save(out_img_path, quality=95, subsampling=0)
    else:
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.png")
        pngInfo = PngImagePlugin.PngInfo()

        if args.write_metadata_to_png:
            pngInfo.add_text(
                "parameters",
                f"{args.prompts[0]}\nNegative prompt: {args.negative_prompts[0]}\nSteps:{args.steps}, Sampler: {args.scheduler}, CFG scale: {args.guidance_scale}, Seed: {img_seed}, Size: {args.width}x{args.height}, Model: {img_model}",
            )

        output_img.save(out_img_path, "PNG", pnginfo=pngInfo)

        if args.output_img_format not in ["png", "jpg"]:
            print(
                f"[ERROR] Format {args.output_img_format} is not supported yet."
                "Image saved as png instead. Supported formats: png / jpg"
            )

    new_entry = {
        "VARIANT": img_model,
        "SCHEDULER": args.scheduler,
        "PROMPT": args.prompts[0],
        "NEG_PROMPT": args.negative_prompts[0],
        "SEED": img_seed,
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


txt2img_obj = None
config_obj = None
schedulers = None


# Exposed to UI.
def txt2img_inf(
    prompt: str,
    negative_prompt: str,
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
    global txt2img_obj
    global config_obj
    global schedulers

    args.prompts = [prompt]
    args.negative_prompts = [negative_prompt]
    args.guidance_scale = guidance_scale
    args.steps = steps
    args.scheduler = scheduler

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
    if config_obj != new_config_obj:
        config_obj = new_config_obj
        args.precision = precision
        args.batch_size = batch_size
        args.max_length = max_length
        args.height = height
        args.width = width
        args.device = device.split("=>", 1)[1].strip()
        args.use_tuned = True
        args.import_mlir = False
        set_init_device_flags()
        model_id = (
            args.hf_model_id
            if args.hf_model_id
            else "stabilityai/stable-diffusion-2-1-base"
        )
        schedulers = get_schedulers(model_id)
        scheduler_obj = schedulers[scheduler]
        txt2img_obj = Text2ImagePipeline.from_pretrained(
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

    if not txt2img_obj:
        sys.exit("text to image pipeline must not return a null value")

    txt2img_obj.scheduler = schedulers[scheduler]

    start_time = time.time()
    txt2img_obj.log = ""
    generated_imgs = []
    seeds = []
    img_seed = utils.sanitize_seed(seed)
    for i in range(batch_count):
        if i > 0:
            img_seed = utils.sanitize_seed(-1)
        out_imgs = txt2img_obj.generate_images(
            prompt,
            negative_prompt,
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
        txt2img_obj.log += "\n"

    total_time = time.time() - start_time
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
    text_output += f"\nscheduler={args.scheduler}, device={device}"
    text_output += f"\nsteps={args.steps}, guidance_scale={args.guidance_scale}, seed={seeds}"
    text_output += f"\nsize={args.height}x{args.width}, batch-count={batch_count}, batch-size={args.batch_size}, max_length={args.max_length}"
    text_output += txt2img_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    return generated_imgs, text_output


if __name__ == "__main__":
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    cpu_scheduling = not args.scheduler.startswith("Shark")
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)
    scheduler_obj = schedulers[args.scheduler]
    seed = args.seed

    txt2img_obj = Text2ImagePipeline.from_pretrained(
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

    for run in range(args.runs):
        if run > 0:
            seed = -1
        seed = utils.sanitize_seed(seed)

        start_time = time.time()
        generated_imgs = txt2img_obj.generate_images(
            args.prompts,
            args.negative_prompts,
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
        # TODO: if using --runs=x txt2img_obj.log will output on each display every iteration infos from the start
        text_output += txt2img_obj.log
        text_output += f"\nTotal image generation time: {total_time:.4f}sec"

        save_output_img(generated_imgs[0], seed)
        print(text_output)
