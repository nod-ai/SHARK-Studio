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

from apps.shark_studio.web.utils.default_configs import default_sd_configs


def write_default_sd_configs(path):
    for key in default_sd_configs.keys():
        config_fpath = os.path.join(path, key)
        with open(config_fpath, "w") as f:
            f.write(default_sd_configs[key])


def safe_name(name):
    return name.split("/")[-1].replace("-", "_")


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
    configs = get_resource_path(cmd_opts.config_dir)
    if not os.path.exists(configs):
        os.mkdir(configs)
    return Path(configs)


def get_generated_imgs_path() -> Path:
    outputs = get_resource_path(cmd_opts.output_dir)
    if not os.path.exists(outputs):
        os.mkdir(outputs)
    return Path(outputs)


def get_tmp_path() -> Path:
    tmpdir = get_resource_path(cmd_opts.model_dir)
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    return Path(tmpdir)


def get_generated_imgs_todays_subdir() -> str:
    return dt.now().strftime("%Y%m%d")


def create_model_folders():
    dir = ["checkpoints", "vae", "lora", "vmfb"]
    if not os.path.isdir(cmd_opts.model_dir):
        try:
            os.makedirs(cmd_opts.model_dir)
        except OSError:
            sys.exit(
                f"Invalid --model_dir argument, "
                f"{cmd_opts.model_dir} folder does not exist, and cannot be created."
            )

    for root in dir:
        Path(get_checkpoints_path(root)).mkdir(parents=True, exist_ok=True)


def get_checkpoints_path(model_type=""):
    return get_resource_path(os.path.join(cmd_opts.model_dir, model_type))


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
