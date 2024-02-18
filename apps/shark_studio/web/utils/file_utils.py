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
