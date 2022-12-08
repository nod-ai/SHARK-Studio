import os

os.environ["AMD_ENABLE_LLPC"] = "1"

import torch
import re
import time
from pathlib import Path
from datetime import datetime as dt
from dataclasses import dataclass
from csv import DictWriter
from apps.stable_diffusion.src import (
    args,
    Text2ImagePipeline,
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
    out_img_path = Path(generated_imgs_path, f"{out_img_name}.jpg")
    output_img.save(out_img_path, quality=95, subsampling=0)

    new_entry = {
        "VARIANT": args.hf_model_id,
        "SCHEDULER": args.scheduler,
        "PROMPT": args.prompts[0],
        "NEG_PROMPT": args.negative_prompts[0],
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
    batch_size: int,
    scheduler: str,
    model_id: str,
    custom_model_id: str,
    ckpt_file_obj,
    precision: str,
    device: str,
    max_length: int,
):
    global txt2img_obj
    global config_obj
    global schedulers

    args.prompts = [prompt]
    args.negative_prompts = [negative_prompt]
    args.guidance_scale = guidance_scale
    args.seed = seed
    args.steps = steps
    args.scheduler = scheduler
    args.hf_model_id = custom_model_id if custom_model_id else model_id
    args.ckpt_loc = ckpt_file_obj.name if ckpt_file_obj else ""
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
        schedulers = get_schedulers(model_id)
        scheduler_obj = schedulers[scheduler]
        txt2img_obj = Text2ImagePipeline.from_pretrained(
            scheduler_obj,
            args.import_mlir,
            args.hf_model_id,
            args.ckpt_loc,
            args.precision,
            args.max_length,
            args.batch_size,
            args.height,
            args.width,
            args.use_base_vae,
        )
    txt2img_obj.scheduler = schedulers[scheduler]

    start_time = time.time()
    txt2img_obj.log = ""
    generated_imgs = txt2img_obj.generate_images(
        prompt,
        negative_prompt,
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
    text_output += txt2img_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    return generated_imgs, text_output


if __name__ == "__main__":
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    cpu_scheduling = not args.scheduler.startswith("Shark")
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)
    scheduler_obj = schedulers[args.scheduler]

    txt2img_obj = Text2ImagePipeline.from_pretrained(
        scheduler_obj,
        args.import_mlir,
        args.hf_model_id,
        args.ckpt_loc,
        args.precision,
        args.max_length,
        args.batch_size,
        args.height,
        args.width,
        args.use_base_vae,
    )

    start_time = time.time()
    generated_imgs = txt2img_obj.generate_images(
        args.prompts,
        args.negative_prompts,
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
    text_output += txt2img_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    save_output_img(generated_imgs[0])
    print(text_output)
