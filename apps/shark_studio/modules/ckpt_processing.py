import os
import json
import re
import requests
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline
from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
    create_vae_diffusers_config,
    convert_ldm_vae_checkpoint,
)


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
    print("Loading diffusers' pipeline from original stable diffusion checkpoint")
    num_in_channels = 9 if is_inpaint else 4
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=custom_weights,
        extract_ema=extract_ema,
        from_safetensors=from_safetensors,
        num_in_channels=num_in_channels,
    )
    pipe.save_pretrained(path_to_diffusers)
    print("Loading complete")

def load_model_from_ckpt(custom_weights, is_inpaint=False):
    path_to_diffusers = get_path_to_diffusers_checkpoint(custom_weights)
    # if next(Path(path_to_diffusers).iterdir(), None):
    #     print("Checkpoint already loaded at : ", path_to_diffusers)
    #     return
    # else:
    #     print(
    #         "Diffusers' checkpoint will be identified here : ",
    #         path_to_diffusers,
    #     )
    from_safetensors = (
        True if custom_weights.lower().endswith(".safetensors") else False
    )
    extract_ema = False
    print("Loading diffusers' pipeline from original stable diffusion checkpoint")
    num_in_channels = 9 if is_inpaint else 4
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict=custom_weights,
        extract_ema=extract_ema,
        from_safetensors=from_safetensors,
        num_in_channels=num_in_channels,
    )
    return pipe


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

    converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae_state_dict, vae_config)
    return converted_vae_checkpoint


def process_custom_pipe_weights(custom_weights):
    if custom_weights != "":
        if custom_weights.startswith("https://civitai.com/api/"):
            # download the checkpoint from civitai if we don't already have it
            weights_path = get_civitai_checkpoint(custom_weights)

            # act as if we were given the local file as custom_weights originally
            custom_weights_tgt = get_path_to_diffusers_checkpoint(weights_path)
            custom_weights_params = weights_path

        else:
            assert custom_weights.lower().endswith(
                (".ckpt", ".safetensors")
            ), "checkpoint files supported can be any of [.ckpt, .safetensors] type"
            custom_weights_tgt = get_path_to_diffusers_checkpoint(custom_weights)
            custom_weights_params = custom_weights

        return custom_weights_params, custom_weights_tgt


def get_civitai_checkpoint(url: str):
    with requests.get(url, allow_redirects=True, stream=True) as response:
        response.raise_for_status()

        # civitai api returns the filename in the content disposition
        base_filename = re.findall(
            '"([^"]*)"', response.headers["Content-Disposition"]
        )[0]
        destination_path = Path.cwd() / (cmd_opts.model_dir or "models") / base_filename

        # we don't have this model downloaded yet
        if not destination_path.is_file():
            print(f"downloading civitai model from {url} to {destination_path}")

            size = int(response.headers["content-length"], 0)
            progress_bar = tqdm(total=size, unit="iB", unit_scale=True)

            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

            progress_bar.close()

        # we already have this model downloaded
        else:
            print(f"civitai model already downloaded to {destination_path}")

        response.close()
        return destination_path.as_posix()
