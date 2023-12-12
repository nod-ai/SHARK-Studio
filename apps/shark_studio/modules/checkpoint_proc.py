import os
import json
import re
from pathlib import Path
from omegaconf import OmegaConf


def get_path_to_diffusers_checkpoint(custom_weights):
    path = Path(custom_weights)
    diffusers_path = path.parent.absolute()
    diffusers_directory_name = os.path.join("diffusers", path.stem)
    complete_path_to_diffusers = diffusers_path / diffusers_directory_name
    complete_path_to_diffusers.mkdir(parents=True, exist_ok=True)
    path_to_diffusers = complete_path_to_diffusers.as_posix()
    return path_to_diffusers


def preprocessCKPT(custom_weights, is_inpaint=False):
    path_to_diffusers = get_path_to_diffusers_checkpoint(custom_weights)
    if next(Path(path_to_diffusers).iterdir(), None):
        print("Checkpoint already loaded at : ", path_to_diffusers)
        return
    else:
        print(
            "Diffusers' checkpoint will be identified here : ",
            path_to_diffusers,
        )
    from_safetensors = (
        True if custom_weights.lower().endswith(".safetensors") else False
    )
    # EMA weights usually yield higher quality images for inference but
    # non-EMA weights have been yielding better results in our case.
    # TODO: Add an option `--ema` (`--no-ema`) for users to specify if
    #  they want to go for EMA weight extraction or not.
    extract_ema = False
    print(
        "Loading diffusers' pipeline from original stable diffusion checkpoint"
    )
    num_in_channels = 9 if is_inpaint else 4
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=custom_weights,
        extract_ema=extract_ema,
        from_safetensors=from_safetensors,
        num_in_channels=num_in_channels,
    )
    pipe.save_pretrained(path_to_diffusers)
    print("Loading complete")


def convert_original_vae(vae_checkpoint):
    vae_state_dict = {}
    for key in list(vae_checkpoint.keys()):
        vae_state_dict["first_stage_model." + key] = vae_checkpoint.get(key)

    config_url = (
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/"
        "main/configs/stable-diffusion/v1-inference.yaml"
    )
    original_config_file = BytesIO(requests.get(config_url).content)
    original_config = OmegaConf.load(original_config_file)
    vae_config = create_vae_diffusers_config(original_config, image_size=512)

    converted_vae_checkpoint = convert_ldm_vae_checkpoint(
        vae_state_dict, vae_config
    )
    return converted_vae_checkpoint
