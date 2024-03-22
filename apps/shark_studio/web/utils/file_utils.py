import os
import sys
import glob
from datetime import datetime as dt
from pathlib import Path

from apps.shark_studio.modules.shared_cmd_opts import cmd_opts

checkpoints_filetypes = (
    "*.ckpt",
    "*.safetensors",
)

default_sd_config = r"""{
  "prompt": [
    "a photo taken of the front of a super-car drifting on a road near mountains at high speeds with smoke coming off the tires, front angle, front point of view, trees in the mountains of the background, ((sharp focus))"
  ],
  "negative_prompt": [
    "watermark, signature, logo, text, lowres, ((monochrome, grayscale)), blurry, ugly, blur, oversaturated, cropped"
  ],
  "sd_init_image": [null],
  "height": 512,
  "width": 512,
  "steps": 50,
  "strength": 0.8,
  "guidance_scale": 7.5,
  "seed": "-1",
  "batch_count": 1,
  "batch_size": 1,
  "scheduler": "EulerDiscrete",
  "base_model_id": "stabilityai/stable-diffusion-2-1-base",
  "custom_weights": null,
  "custom_vae": null,
  "precision": "fp16",
  "device": "AMD Radeon RX 7900 XTX => vulkan://0",
  "ondemand": false,
  "repeatable_seeds": false,
  "resample_type": "Nearest Neighbor",
  "controlnets": {},
  "embeddings": {}
}"""

def write_default_sd_config(path):
    with open(path, "w") as f:
        f.write(default_sd_config)

def safe_name(name):
    return name.replace("/", "_").replace("-", "_")


def get_path_stem(path):
    path = Path(path)
    return path.stem


def get_resource_path(path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if os.path.isabs(path):
        return path
    else:
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        result = Path(os.path.join(base_path, path)).resolve(strict=False)
        return result


def get_configs_path() -> Path:
    configs = get_resource_path(os.path.join("..", "configs"))
    if not os.path.exists(configs):
        os.mkdir(configs)
    return Path(get_resource_path("../configs"))


def get_generated_imgs_path() -> Path:
    return Path(
        cmd_opts.output_dir
        if cmd_opts.output_dir
        else get_resource_path("../generated_imgs")
    )


def get_generated_imgs_todays_subdir() -> str:
    return dt.now().strftime("%Y%m%d")


def create_checkpoint_folders():
    dir = ["checkpoints", "vae", "lora", "vmfb"]
    if not os.path.isdir(cmd_opts.ckpt_dir):
        try:
            os.makedirs(cmd_opts.ckpt_dir)
        except OSError:
            sys.exit(
                f"Invalid --ckpt_dir argument, "
                f"{cmd_opts.ckpt_dir} folder does not exist, and cannot be created."
            )

    for root in dir:
        Path(get_checkpoints_path(root)).mkdir(parents=True, exist_ok=True)


def get_checkpoints_path(model_type=""):
    return get_resource_path(os.path.join(cmd_opts.ckpt_dir, model_type))


def get_checkpoints(model_type="checkpoints"):
    ckpt_files = []
    file_types = checkpoints_filetypes
    if model_type == "lora":
        file_types = file_types + ("*.pt", "*.bin")
    for extn in file_types:
        files = [
            os.path.basename(x)
            for x in glob.glob(os.path.join(get_checkpoints_path(model_type), extn))
        ]
    ckpt_files.extend(files)
    return sorted(ckpt_files, key=str.casefold)


def get_checkpoint_pathfile(checkpoint_name, model_type="checkpoints"):
    return os.path.join(get_checkpoints_path(model_type), checkpoint_name)
