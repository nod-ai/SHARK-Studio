import os
import sys
import torch
import json
import safetensors
from safetensors.torch import load_file
from apps.shark_studio.api.utils import get_checkpoint_pathfile


def processLoRA(model, use_lora, splitting_prefix):
    state_dict = ""
    if ".safetensors" in use_lora:
        state_dict = load_file(use_lora)
    else:
        state_dict = torch.load(use_lora)
    alpha = 0.75
    visited = []

    # directly update weight in model
    process_unet = "te" not in splitting_prefix
    for key in state_dict:
        if ".alpha" in key or key in visited:
            continue

        curr_layer = model
        if ("text" not in key and process_unet) or (
            "text" in key and not process_unet
        ):
            layer_infos = (
                key.split(".")[0].split(splitting_prefix)[-1].split("_")
            )
        else:
            continue

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = (
                state_dict[pair_keys[0]]
                .squeeze(3)
                .squeeze(2)
                .to(torch.float32)
            )
            weight_down = (
                state_dict[pair_keys[1]]
                .squeeze(3)
                .squeeze(2)
                .to(torch.float32)
            )
            curr_layer.weight.data += alpha * torch.mm(
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
        # update visited list
        for item in pair_keys:
            visited.append(item)
    return model


def update_lora_weight_for_unet(unet, use_lora):
    extensions = [".bin", ".safetensors", ".pt"]
    if not any([extension in use_lora for extension in extensions]):
        # We assume if it is a HF ID with standalone LoRA weights.
        unet.load_attn_procs(use_lora)
        return unet

    main_file_name = get_path_stem(use_lora)
    if ".bin" in use_lora:
        main_file_name += ".bin"
    elif ".safetensors" in use_lora:
        main_file_name += ".safetensors"
    elif ".pt" in use_lora:
        main_file_name += ".pt"
    else:
        sys.exit("Only .bin and .safetensors format for LoRA is supported")

    try:
        dir_name = os.path.dirname(use_lora)
        unet.load_attn_procs(dir_name, weight_name=main_file_name)
        return unet
    except:
        return processLoRA(unet, use_lora, "lora_unet_")


def update_lora_weight(model, use_lora, model_name):
    if "unet" in model_name:
        return update_lora_weight_for_unet(model, use_lora)
    try:
        return processLoRA(model, use_lora, "lora_te_")
    except:
        return None


def get_lora_metadata(lora_filename):
    # get the metadata from the file
    filename = get_checkpoint_pathfile(lora_filename, "lora")
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
