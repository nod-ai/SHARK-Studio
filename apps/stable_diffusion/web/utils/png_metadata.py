import re
from pathlib import Path
from apps.stable_diffusion.web.ui.utils import (
    get_custom_model_pathfile,
    scheduler_list_txt2img,
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
    hf_model_id,
):
    try:
        png_info = pil_data.info["parameters"]
        metadata = parse_generation_parameters(png_info)
        png_hf_model_id = ""
        png_custom_model = ""

        if "Model" in metadata:
            # Remove extension from model info
            if metadata["Model"].endswith(".safetensors") or metadata[
                "Model"
            ].endswith(".ckpt"):
                metadata["Model"] = Path(metadata["Model"]).stem
            # Check for the model name match with one of the local ckpt or safetensors files
            if Path(
                get_custom_model_pathfile(metadata["Model"] + ".ckpt")
            ).is_file():
                png_custom_model = metadata["Model"] + ".ckpt"
            if Path(
                get_custom_model_pathfile(metadata["Model"] + ".safetensors")
            ).is_file():
                png_custom_model = metadata["Model"] + ".safetensors"
            # Check for a model match with one of the default model list (ex: "Linaqruf/anything-v3.0")
            if metadata["Model"] in predefined_models:
                png_custom_model = metadata["Model"]
            # If nothing had matched, check vendor/hf_model_id
            if not png_custom_model and metadata["Model"].count("/"):
                png_hf_model_id = metadata["Model"]
            # No matching model was found
            if not png_custom_model and not png_hf_model_id:
                print(
                    "Import PNG info: Unable to find a matching model for %s"
                    % metadata["Model"]
                )

        negative_prompt = metadata["Negative prompt"]
        steps = int(metadata["Steps"])
        cfg_scale = float(metadata["CFG scale"])
        seed = int(metadata["Seed"])
        width = float(metadata["Size-1"])
        height = float(metadata["Size-2"])
        if "Model" in metadata and png_custom_model:
            custom_model = png_custom_model
            hf_model_id = ""
        if "Model" in metadata and png_hf_model_id:
            custom_model = "None"
            hf_model_id = png_hf_model_id
        if "Prompt" in metadata:
            prompt = metadata["Prompt"]
        if "Sampler" in metadata:
            if metadata["Sampler"] in scheduler_list_txt2img:
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
        hf_model_id,
    )
