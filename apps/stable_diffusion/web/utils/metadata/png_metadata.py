import re
from pathlib import Path
from apps.stable_diffusion.web.ui.utils import (
    get_custom_model_pathfile,
    scheduler_list,
    predefined_models,
)

re_param_code = r'\s*([\w ]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")


def parse_generation_parameters(x: str):
    res = {}
    prompt = ""
    negative_prompt = ""
    done_with_prompt = False

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ""

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()

        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt

    for k, v in re_param.findall(lastline):
        v = v[1:-1] if v[0] == '"' and v[-1] == '"' else v
        m = re_imagesize.match(v)
        if m is not None:
            res[k + "-1"] = m.group(1)
            res[k + "-2"] = m.group(2)
        else:
            res[k] = v

    # Missing CLIP skip means it was set to 1 (the default)
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    hypernet = res.get("Hypernet", None)
    if hypernet is not None:
        res[
            "Prompt"
        ] += f"""<hypernet:{hypernet}:{res.get("Hypernet strength", "1.0")}>"""

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    return res


def try_find_model_base_from_png_metadata(
    file: str, folder: str = "models"
) -> str:
    custom = ""

    # Remove extension from file info
    if file.endswith(".safetensors") or file.endswith(".ckpt"):
        file = Path(file).stem
    # Check for the file name match with one of the local ckpt or safetensors files
    if Path(get_custom_model_pathfile(file + ".ckpt", folder)).is_file():
        custom = file + ".ckpt"
    if Path(
        get_custom_model_pathfile(file + ".safetensors", folder)
    ).is_file():
        custom = file + ".safetensors"

    return custom


def find_model_from_png_metadata(
    key: str, metadata: dict[str, str | int]
) -> tuple[str, str]:
    png_hf_id = ""
    png_custom = ""

    if key in metadata:
        model_file = metadata[key]
        png_custom = try_find_model_base_from_png_metadata(model_file)
        # Check for a model match with one of the default model list (ex: "Linaqruf/anything-v3.0")
        if model_file in predefined_models:
            png_custom = model_file
        # If nothing had matched, check vendor/hf_model_id
        if not png_custom and model_file.count("/"):
            png_hf_id = model_file
        # No matching model was found
        if not png_custom and not png_hf_id:
            print(
                "Import PNG info: Unable to find a matching model for %s"
                % model_file
            )

    return png_custom, png_hf_id


def find_vae_from_png_metadata(
    key: str, metadata: dict[str, str | int]
) -> str:
    vae_custom = ""

    if key in metadata:
        vae_file = metadata[key]
        vae_custom = try_find_model_base_from_png_metadata(vae_file, "vae")

    # VAE input is optional, should not print or throw an error if missing

    return vae_custom


def find_lora_from_png_metadata(
    key: str, metadata: dict[str, str | int]
) -> tuple[str, str]:
    lora_custom = ""

    if key in metadata:
        lora_file = metadata[key]
        lora_custom = try_find_model_base_from_png_metadata(lora_file, "lora")
        # If nothing had matched, check vendor/hf_model_id
        if not lora_custom and lora_file.count("/"):
            lora_custom = lora_file

    # LoRA input is optional, should not print or throw an error if missing
    return lora_custom


def import_png_metadata(
    pil_data,
    prompt,
    negative_prompt,
    steps,
    sampler,
    cfg_scale,
    seed,
    width,
    height,
    custom_model,
    custom_lora,
    custom_vae,
):
    try:
        png_info = pil_data.info["parameters"]
        metadata = parse_generation_parameters(png_info)

        (png_custom_model, png_hf_model_id) = find_model_from_png_metadata(
            "Model", metadata
        )
        lora_custom_model = find_lora_from_png_metadata("LoRA", metadata)
        vae_custom_model = find_vae_from_png_metadata("VAE", metadata)

        negative_prompt = metadata["Negative prompt"]
        steps = int(metadata["Steps"])
        cfg_scale = float(metadata["CFG scale"])
        seed = int(metadata["Seed"])
        width = float(metadata["Size-1"])
        height = float(metadata["Size-2"])

        if "Model" in metadata and png_custom_model:
            custom_model = png_custom_model
        elif "Model" in metadata and png_hf_model_id:
            custom_model = png_hf_model_id

        if "LoRA" in metadata and lora_custom_model:
            custom_lora = lora_custom_model
        else:
            custom_lora = "None"

        if "VAE" in metadata and vae_custom_model:
            custom_vae = vae_custom_model

        if "Prompt" in metadata:
            prompt = metadata["Prompt"]
        if "Sampler" in metadata:
            if metadata["Sampler"] in scheduler_list:
                sampler = metadata["Sampler"]
            else:
                print(
                    "Import PNG info: Unable to find a scheduler for %s"
                    % metadata["Sampler"]
                )

    except Exception as ex:
        if pil_data and pil_data.info.get("parameters"):
            print("import_png_metadata failed with %s" % ex)
        pass

    return (
        None,
        prompt,
        negative_prompt,
        steps,
        sampler,
        cfg_scale,
        seed,
        width,
        height,
        custom_model,
        custom_lora,
        custom_vae,
    )
