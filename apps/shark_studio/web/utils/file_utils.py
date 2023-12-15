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


def safe_name(name):
    return name.replace("/", "_").replace("-", "_")


def get_path_stem(path):
    path = Path(path)
    return path.stem


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    result = Path(os.path.join(base_path, relative_path)).resolve(strict=False)
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
    dir = ["vae", "lora"]
    if not cmd_opts.ckpt_dir:
        dir.insert(0, "models")
    else:
        if not os.path.isdir(cmd_opts.ckpt_dir):
            sys.exit(
                f"Invalid --ckpt_dir argument, "
                f"{cmd_opts.ckpt_dir} folder does not exists."
            )
    for root in dir:
        Path(get_checkpoints_path(root)).mkdir(parents=True, exist_ok=True)


def get_checkpoints_path(model=""):
    return get_resource_path(f"../models/{model}")


def get_checkpoints(model="models"):
    ckpt_files = []
    file_types = checkpoints_filetypes
    if model == "lora":
        file_types = file_types + ("*.pt", "*.bin")
    for extn in file_types:
        files = [
            os.path.basename(x)
            for x in glob.glob(os.path.join(get_checkpoints_path(model), extn))
        ]
    ckpt_files.extend(files)
    return sorted(ckpt_files, key=str.casefold)


def get_checkpoint_pathfile(checkpoint_name, model="models"):
    return os.path.join(get_checkpoints_path(model), checkpoint_name)
