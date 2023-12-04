import os
import sys
import glob
import math
import json
import safetensors
import gradio as gr

from pathlib import Path
from apps.stable_diffusion.src import args
from dataclasses import dataclass
from enum import IntEnum

from apps.stable_diffusion.src import get_available_devices
import apps.stable_diffusion.web.utils.global_obj as global_obj
from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
    SD_STATE_CANCEL,
)


@dataclass
class Config:
    mode: str
    model_id: str
    ckpt_loc: str
    custom_vae: str
    precision: str
    batch_size: int
    max_length: int
    height: int
    width: int
    device: str
    use_lora: str
    stencils: list[str]
    ondemand: str  # should this be expecting a bool instead?


class HSLHue(IntEnum):
    RED = 0
    YELLOW = 60
    GREEN = 120
    CYAN = 180
    BLUE = 240
    MAGENTA = 300


custom_model_filetypes = (
    "*.ckpt",
    "*.safetensors",
)  # the tuple of file types

scheduler_list_cpu_only = [
    "DDIM",
    "PNDM",
    "LMSDiscrete",
    "KDPM2Discrete",
    "DPMSolverMultistep",
    "DPMSolverMultistep++",
    "DPMSolverMultistepKarras",
    "DPMSolverMultistepKarras++",
    "EulerDiscrete",
    "EulerAncestralDiscrete",
    "DEISMultistep",
    "KDPM2AncestralDiscrete",
    "DPMSolverSinglestep",
    "DDPM",
    "HeunDiscrete",
    "LCMScheduler",
]
scheduler_list = scheduler_list_cpu_only + [
    "SharkEulerDiscrete",
    "SharkEulerAncestralDiscrete",
]

predefined_models = [
    "Linaqruf/anything-v3.0",
    "prompthero/openjourney",
    "wavymulder/Analog-Diffusion",
    "xzuyn/PhotoMerge",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base",
    "CompVis/stable-diffusion-v1-4",
]

predefined_paint_models = [
    "runwayml/stable-diffusion-inpainting",
    "stabilityai/stable-diffusion-2-inpainting",
    "xzuyn/PhotoMerge-inpainting",
]
predefined_upscaler_models = [
    "stabilityai/stable-diffusion-x4-upscaler",
]
predefined_sdxl_models = [
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-xl-base-1.0",
]


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_path, relative_path)


def create_custom_models_folders():
    dir = ["vae", "lora"]
    if not args.ckpt_dir:
        dir.insert(0, "models")
    else:
        if not os.path.isdir(args.ckpt_dir):
            sys.exit(
                f"Invalid --ckpt_dir argument, "
                f"{args.ckpt_dir} folder does not exists."
            )
    for root in dir:
        get_custom_model_path(root).mkdir(parents=True, exist_ok=True)


def get_custom_model_path(model="models"):
    # structure in WebUI :-
    #       models or args.ckpt_dir
    #         |___lora
    #         |___vae
    sub_folder = "" if model == "models" else model
    if args.ckpt_dir:
        return Path(Path(args.ckpt_dir), sub_folder)
    else:
        return Path(Path.cwd(), "models/" + sub_folder)


def get_custom_model_pathfile(custom_model_name, model="models"):
    return os.path.join(get_custom_model_path(model), custom_model_name)


def get_custom_model_files(model="models", custom_checkpoint_type=""):
    ckpt_files = []
    file_types = custom_model_filetypes
    if model == "lora":
        file_types = custom_model_filetypes + ("*.pt", "*.bin")
    for extn in file_types:
        files = [
            os.path.basename(x)
            for x in glob.glob(
                os.path.join(get_custom_model_path(model), extn)
            )
        ]
        match custom_checkpoint_type:
            case "sdxl":
                files = [
                    val
                    for val in files
                    if any(x in val for x in ["XL", "xl", "Xl"])
                ]
            case "inpainting":
                files = [
                    val
                    for val in files
                    if val.endswith("inpainting" + extn.removeprefix("*"))
                ]
            case "upscaler":
                files = [
                    val
                    for val in files
                    if val.endswith("upscaler" + extn.removeprefix("*"))
                ]
            case _:
                files = [
                    val
                    for val in files
                    if not (
                        val.endswith("inpainting" + extn.removeprefix("*"))
                        or val.endswith("upscaler" + extn.removeprefix("*"))
                    )
                ]
        ckpt_files.extend(files)
    return sorted(ckpt_files, key=str.casefold)


def get_custom_vae_or_lora_weights(weights, hf_id, model):
    use_weight = ""
    if weights == "None" and not hf_id:
        use_weight = ""
    elif not hf_id:
        use_weight = get_custom_model_pathfile(weights, model)
    else:
        use_weight = hf_id
    return use_weight


def hsl_color(alpha: float, start, end):
    b = (end - start) * (alpha if alpha > 0 else 0)
    result = b + start

    # Return a CSS HSL string
    return f"hsl({math.floor(result)}, 80%, 35%)"


def get_lora_metadata(lora_filename):
    # get the metadata from the file
    filename = get_custom_model_pathfile(lora_filename, "lora")
    with safetensors.safe_open(filename, framework="pt", device="cpu") as f:
        metadata = f.metadata()

    # guard clause for if there isn't any metadata
    if not metadata:
        return None

    # metadata is a dictionary of strings, the values of the keys we're
    # interested in are actually json, and need to be loaded as such
    tag_frequencies = json.loads(metadata.get("ss_tag_frequency", str("{}")))
    dataset_dirs = json.loads(metadata.get("ss_dataset_dirs", str("{}")))
    tag_dirs = [dir for dir in tag_frequencies.keys()]

    # gather the tag frequency information for all the datasets trained
    all_frequencies = {}
    for dataset in tag_dirs:
        frequencies = sorted(
            [entry for entry in tag_frequencies[dataset].items()],
            reverse=True,
            key=lambda x: x[1],
        )

        # get a figure for the total number of images processed for this dataset
        # either then number actually listed or in its dataset_dir entry or
        # the highest frequency's number if that doesn't exist
        img_count = dataset_dirs.get(dir, {}).get(
            "img_count", frequencies[0][1]
        )

        # add the dataset frequencies to the overall frequencies replacing the
        # frequency counts on the tags with a percentage/ratio
        all_frequencies.update(
            [(entry[0], entry[1] / img_count) for entry in frequencies]
        )

    trained_model_id = " ".join(
        [
            metadata.get("ss_sd_model_hash", ""),
            metadata.get("ss_sd_model_name", ""),
            metadata.get("ss_base_model_version", ""),
        ]
    ).strip()

    # return the topmost <count> of all frequencies in all datasets
    return {
        "model": trained_model_id,
        "frequencies": sorted(
            all_frequencies.items(), reverse=True, key=lambda x: x[1]
        ),
    }


def cancel_sd():
    # Try catch it, as gc can delete global_obj.sd_obj while switching model
    try:
        global_obj.set_sd_status(SD_STATE_CANCEL)
    except Exception:
        pass


def set_model_default_configs(model_ckpt_or_id, jsonconfig=None):
    import gradio as gr

    if jsonconfig:
        return get_config_from_json(jsonconfig)
    elif default_config_exists(model_ckpt_or_id):
        return default_configs[model_ckpt_or_id]
    # TODO: Use HF metadata to setup pipeline if available
    # elif is_valid_hf_id(model_ckpt_or_id):
    #     return get_HF_default_configs(model_ckpt_or_id)
    else:
        # We don't have default metadata to setup a good config. Do not change configs.
        return [
            gr.Textbox(label="Prompt", interactive=True, visible=True),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        ]


def get_config_from_json(model_ckpt_or_id, jsonconfig):
    # TODO: make this work properly. It is currently not user-exposed.
    cfgdata = json.load(jsonconfig)
    return [
        cfgdata["prompt_box_behavior"],
        cfgdata["neg_prompt_box_behavior"],
        cfgdata["steps"],
        cfgdata["scheduler"],
        cfgdata["guidance_scale"],
        cfgdata["width"],
        cfgdata["height"],
        cfgdata["custom_vae"],
    ]


def default_config_exists(model_ckpt_or_id):
    if model_ckpt_or_id in [
        "stabilityai/sdxl-turbo",
        "stabilityai/stable_diffusion-xl-base-1.0",
    ]:
        return True
    else:
        return False


default_configs = {
    "stabilityai/sdxl-turbo": [
        gr.Textbox(label="", interactive=False, value=None, visible=False),
        gr.Textbox(
            label="Prompt",
            value="An anthropomorphic shark writing code on an old tube monitor, macro shot, in an office filled with water, stop-animation style, claymation",
        ),
        gr.Slider(0, 5, value=2),
        gr.Dropdown(value="EulerAncestralDiscrete"),
        gr.Slider(0, value=0),
        512,
        512,
        "madebyollin/sdxl-vae-fp16-fix",
    ],
    "stabilityai/stable-diffusion-xl-base-1.0": [
        gr.Textbox(label="Prompt", interactive=True, visible=True),
        gr.Textbox(label="Negative Prompt", interactive=True),
        40,
        "EulerDiscrete",
        7.5,
        gr.Slider(value=1024, interactive=False),
        gr.Slider(value=1024, interactive=False),
        "madebyollin/sdxl-vae-fp16-fix",
    ],
}

nodlogo_loc = resource_path("logos/nod-logo.png")
nodicon_loc = resource_path("logos/nod-icon.png")
available_devices = get_available_devices()
