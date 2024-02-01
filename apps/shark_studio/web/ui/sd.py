import os
import json
import gradio as gr
import numpy as np
from inspect import signature
from PIL import Image
from pathlib import Path
from datetime import datetime as dt
from gradio.components.image_editor import (
    EditorValue,
)
from apps.shark_studio.web.utils.file_utils import (
    get_generated_imgs_path,
    get_checkpoints_path,
    get_checkpoints,
    get_configs_path,
)
from apps.shark_studio.api.sd import (
    sd_model_map,
    shark_sd_fn_dict_input,
    cancel_sd,
)
from apps.shark_studio.api.controlnet import (
    cnet_preview,
)
from apps.shark_studio.modules.schedulers import (
    scheduler_model_map,
)
from apps.shark_studio.modules.img_processing import (
    resampler_list,
    resize_stencil,
)
from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
from apps.shark_studio.web.ui.utils import (
    nodlogo_loc,
    none_to_str_none,
    str_none_to_none,
)
from apps.shark_studio.web.utils.state import (
    status_label,
)
from apps.shark_studio.web.ui.common_events import lora_changed
from apps.shark_studio.modules import logger
import apps.shark_studio.web.utils.globals as global_obj

sd_default_models = [
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-1.0",
    "stabilityai/sdxl-turbo",
]


def view_json_file(file_path):
    content = ""
    with open(file_path, "r") as fopen:
        content = fopen.read()
    return content


def submit_to_cnet_config(
    stencil: str,
    preprocessed_hint: str,
    cnet_strength: int,
    control_mode: str,
    curr_config: dict,
):
    if any(i in [None, ""] for i in [stencil, preprocessed_hint]):
        return gr.update()
    if curr_config is not None:
        if "controlnets" in curr_config:
            curr_config["controlnets"]["control_mode"] = control_mode
            curr_config["controlnets"]["model"].append(stencil)
            curr_config["controlnets"]["hint"].append(preprocessed_hint)
            curr_config["controlnets"]["strength"].append(cnet_strength)
            return curr_config

    cnet_map = {}
    cnet_map["controlnets"] = {
        "control_mode": control_mode,
        "model": [stencil],
        "hint": [preprocessed_hint],
        "strength": [cnet_strength],
    }
    return cnet_map


def update_embeddings_json(embedding):
    return {"embeddings": [embedding]}


def submit_to_main_config(input_cfg: dict, main_cfg: dict):
    if main_cfg in [None, "", {}]:
        return input_cfg

    for base_key in input_cfg:
        main_cfg[base_key] = input_cfg[base_key]
    return main_cfg


def pull_sd_configs(
    prompt,
    negative_prompt,
    sd_init_image,
    height,
    width,
    steps,
    strength,
    guidance_scale,
    seed,
    batch_count,
    batch_size,
    scheduler,
    base_model_id,
    custom_weights,
    custom_vae,
    precision,
    device,
    ondemand,
    repeatable_seeds,
    resample_type,
    controlnets,
    embeddings,
):
    sd_args = str_none_to_none(locals())
    sd_cfg = {}
    for arg in sd_args:
        if arg in [
            "prompt",
            "negative_prompt",
            "sd_init_image",
        ]:
            sd_cfg[arg] = [sd_args[arg]]
        elif arg in ["controlnets", "embeddings"]:
            if isinstance(arg, dict):
                sd_cfg[arg] = json.loads(sd_args[arg])
            else:
                sd_cfg[arg] = {}
        else:
            sd_cfg[arg] = sd_args[arg]

    return json.dumps(sd_cfg)


def load_sd_cfg(sd_json: dict, load_sd_config: str):
    new_sd_config = none_to_str_none(json.loads(view_json_file(load_sd_config)))
    if sd_json:
        for key in new_sd_config:
            sd_json[key] = new_sd_config[key]
    else:
        sd_json = new_sd_config
    for i in sd_json["sd_init_image"]:
        if i is not None:
            if os.path.isfile(i):
                sd_image = [Image.open(i, mode="r")]
    else:
        sd_image = None

    return [
        sd_json["prompt"][0],
        sd_json["negative_prompt"][0],
        sd_image,
        sd_json["height"],
        sd_json["width"],
        sd_json["steps"],
        sd_json["strength"],
        sd_json["guidance_scale"],
        sd_json["seed"],
        sd_json["batch_count"],
        sd_json["batch_size"],
        sd_json["scheduler"],
        sd_json["base_model_id"],
        sd_json["custom_weights"],
        sd_json["custom_vae"],
        sd_json["precision"],
        sd_json["device"],
        sd_json["ondemand"],
        sd_json["repeatable_seeds"],
        sd_json["resample_type"],
        sd_json["controlnets"],
        sd_json["embeddings"],
        sd_json,
    ]


def save_sd_cfg(config: dict, save_name: str):
    if os.path.exists(save_name):
        filepath = save_name
    elif cmd_opts.configs_path:
        filepath = os.path.join(cmd_opts.configs_path, save_name)
    else:
        filepath = os.path.join(get_configs_path(), save_name)
    if ".json" not in filepath:
        filepath += ".json"
    with open(filepath, mode="w") as f:
        f.write(json.dumps(config))
    return "..."


def create_canvas(width, height):
    data = Image.fromarray(
        np.zeros(
            shape=(height, width, 3),
            dtype=np.uint8,
        )
        + 255
    )
    img_dict = {
        "background": data,
        "layers": [],
        "composite": None,
    }
    return EditorValue(img_dict)


def import_original(original_img, width, height):
    if original_img is None:
        resized_img = create_canvas(width, height)
        return resized_img
    else:
        resized_img, _, _ = resize_stencil(original_img, width, height)
        img_dict = {
            "background": resized_img,
            "layers": [],
            "composite": None,
        }
        return EditorValue(img_dict)


with gr.Blocks(title="Stable Diffusion") as sd_element:
    with gr.Row(elem_id="ui_title"):
        nod_logo = Image.open(nodlogo_loc)
        with gr.Row(variant="compact", equal_height=True):
            with gr.Column(
                scale=1,
                elem_id="demo_title_outer",
            ):
                gr.Image(
                    value=nod_logo,
                    show_label=False,
                    interactive=False,
                    elem_id="top_logo",
                    width=150,
                    height=50,
                    show_download_button=False,
                )
    with gr.Column(elem_id="ui_body"):
        with gr.Row():
            with gr.Column(scale=2, min_width=600):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        sd_model_info = (
                            f"Checkpoint Path: {str(get_checkpoints_path())}"
                        )
                        base_model_id = gr.Dropdown(
                            label="Base Model",
                            info="Select or enter HF model ID",
                            elem_id="custom_model",
                            value="stabilityai/stable-diffusion-2-1-base",
                            choices=sd_default_models,
                        )  # base_model_id
                        custom_weights = gr.Dropdown(
                            label="Custom Weights",
                            info="Select or enter HF model ID",
                            elem_id="custom_model",
                            value="None",
                            allow_custom_value=True,
                            choices=["None"] + get_checkpoints(base_model_id),
                        )  #
                    with gr.Column(scale=2):
                        sd_vae_info = (str(get_checkpoints_path("vae"))).replace(
                            "\\", "\n\\"
                        )
                        sd_vae_info = f"VAE Path: {sd_vae_info}"
                        custom_vae = gr.Dropdown(
                            label=f"Custom VAE Models",
                            info=sd_vae_info,
                            elem_id="custom_model",
                            value=(
                                os.path.basename(cmd_opts.custom_vae)
                                if cmd_opts.custom_vae
                                else "None"
                            ),
                            choices=["None"] + get_checkpoints("vae"),
                            allow_custom_value=True,
                            scale=1,
                        )
                        with gr.Row():
                            ondemand = gr.Checkbox(
                                value=cmd_opts.lowvram,
                                label="Low VRAM",
                                interactive=True,
                            )
                            precision = gr.Radio(
                                label="Precision",
                                value=cmd_opts.precision,
                                choices=[
                                    "fp16",
                                    "fp32",
                                ],
                                visible=True,
                            )
                with gr.Group(elem_id="prompt_box_outer"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=cmd_opts.prompt[0],
                        lines=2,
                        elem_id="prompt_box",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=cmd_opts.negative_prompt[0],
                        lines=2,
                        elem_id="negative_prompt_box",
                    )

                with gr.Accordion(label="Input Image", open=False):
                    # TODO: make this import image prompt info if it exists
                    sd_init_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=300,
                        interactive=True,
                    )
                with gr.Accordion(label="Embeddings options", open=True, render=True):
                    sd_lora_info = (str(get_checkpoints_path("loras"))).replace(
                        "\\", "\n\\"
                    )
                    with gr.Row():
                        embeddings_config = gr.JSON(min_width=50, scale=1)
                        lora_opt = gr.Dropdown(
                            allow_custom_value=True,
                            label=f"Standalone LoRA Weights",
                            info=sd_lora_info,
                            elem_id="lora_weights",
                            value=None,
                            multiselect=True,
                            choices=[] + get_checkpoints("lora"),
                            scale=2,
                        )
                        lora_tags = gr.HTML(
                            value="<div><i>No LoRA selected</i></div>",
                            elem_classes="lora-tags",
                        )
                    gr.on(
                        triggers=[lora_opt.change],
                        fn=lora_changed,
                        inputs=[lora_opt],
                        outputs=[lora_tags],
                        queue=True,
                        show_progress=False,
                    ).then(
                        fn=update_embeddings_json,
                        inputs=[lora_opt],
                        outputs=[embeddings_config],
                        show_progress=False,
                    )
                with gr.Accordion(label="Advanced Options", open=True):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            elem_id="scheduler",
                            label="Scheduler",
                            value="EulerDiscrete",
                            choices=scheduler_model_map.keys(),
                            allow_custom_value=False,
                        )
                        height = gr.Slider(
                            384,
                            768,
                            value=cmd_opts.height,
                            step=8,
                            label="Height",
                        )
                        width = gr.Slider(
                            384,
                            768,
                            value=cmd_opts.width,
                            step=8,
                            label="Width",
                        )
                    with gr.Row():
                        with gr.Column(scale=3):
                            steps = gr.Slider(
                                1,
                                100,
                                value=cmd_opts.steps,
                                step=1,
                                label="Steps",
                            )
                            batch_count = gr.Slider(
                                1,
                                100,
                                value=cmd_opts.batch_count,
                                step=1,
                                label="Batch Count",
                                interactive=True,
                            )
                            batch_size = gr.Slider(
                                1,
                                4,
                                value=cmd_opts.batch_size,
                                step=1,
                                label="Batch Size",
                                interactive=True,
                                visible=True,
                            )
                            repeatable_seeds = gr.Checkbox(
                                cmd_opts.repeatable_seeds,
                                label="Repeatable Seeds",
                            )
                        with gr.Column(scale=3):
                            strength = gr.Slider(
                                0,
                                1,
                                value=cmd_opts.strength,
                                step=0.01,
                                label="Denoising Strength",
                            )
                            resample_type = gr.Dropdown(
                                value=cmd_opts.resample_type,
                                choices=resampler_list,
                                label="Resample Type",
                                allow_custom_value=True,
                            )
                            guidance_scale = gr.Slider(
                                0,
                                50,
                                value=cmd_opts.guidance_scale,
                                step=0.1,
                                label="CFG Scale",
                            )
                with gr.Row():
                    seed = gr.Textbox(
                        value=cmd_opts.seed,
                        label="Seed",
                        info="An integer or a JSON list of integers, -1 for random",
                    )
                    device = gr.Dropdown(
                        elem_id="device",
                        label="Device",
                        value=global_obj.get_device_list()[0],
                        choices=global_obj.get_device_list(),
                        allow_custom_value=False,
                    )
                with gr.Accordion(
                    label="Controlnet Options",
                    open=False,
                    visible=False,
                ):
                    preprocessed_hints = gr.State([])
                    with gr.Column():
                        sd_cnet_info = (
                            str(get_checkpoints_path("controlnet"))
                        ).replace("\\", "\n\\")
                    with gr.Row():
                        cnet_config = gr.JSON()
                        with gr.Column():
                            clear_config = gr.ClearButton(
                                value="Clear Controlnet Config",
                                size="sm",
                                components=cnet_config,
                            )
                            control_mode = gr.Radio(
                                choices=["Prompt", "Balanced", "Controlnet"],
                                value="Balanced",
                                label="Control Mode",
                            )
                    with gr.Row():
                        with gr.Column(scale=1):
                            cnet_model = gr.Dropdown(
                                allow_custom_value=True,
                                label=f"Controlnet Model",
                                info=sd_cnet_info,
                                value="None",
                                choices=[
                                    "None",
                                    "canny",
                                    "openpose",
                                    "scribble",
                                    "zoedepth",
                                ]
                                + get_checkpoints("controlnet"),
                            )
                            cnet_strength = gr.Slider(
                                label="Controlnet Strength",
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=1,
                            )
                            with gr.Row():
                                canvas_width = gr.Slider(
                                    label="Canvas Width",
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=8,
                                )
                                canvas_height = gr.Slider(
                                    label="Canvas Height",
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=8,
                                )
                            make_canvas = gr.Button(
                                value="Make Canvas!",
                            )
                            use_input_img = gr.Button(
                                value="Use Original Image",
                                size="sm",
                            )
                        cnet_input = gr.Image(
                            value=None,
                            type="pil",
                            image_mode="RGB",
                            interactive=True,
                        )
                        with gr.Column(scale=1):
                            cnet_output = gr.Image(
                                value=None,
                                visible=True,
                                label="Preprocessed Hint",
                                interactive=False,
                                show_label=True,
                            )
                            cnet_gen = gr.Button(
                                value="Preprocess controlnet input",
                            )
                            use_result = gr.Button(
                                "Submit",
                                size="sm",
                            )
                        use_input_img.click(
                            fn=import_original,
                            inputs=[
                                sd_init_image,
                                canvas_width,
                                canvas_height,
                            ],
                            outputs=[cnet_input],
                            queue=False,
                        )
                        make_canvas.click(
                            fn=create_canvas,
                            inputs=[canvas_width, canvas_height],
                            outputs=[cnet_input],
                            queue=False,
                        )
                        cnet_gen.click(
                            fn=cnet_preview,
                            inputs=[
                                cnet_model,
                                cnet_input,
                            ],
                            outputs=[
                                cnet_output,
                                preprocessed_hints,
                            ],
                        )
                        use_result.click(
                            fn=submit_to_cnet_config,
                            inputs=[
                                cnet_model,
                                cnet_output,
                                cnet_strength,
                                control_mode,
                                cnet_config,
                            ],
                            outputs=[
                                cnet_config,
                            ],
                            queue=False,
                        )
            with gr.Column(scale=3, min_width=600):
                with gr.Group():
                    sd_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                        columns=2,
                        object_fit="fit",
                        preview=True,
                    )
                    std_output = gr.Textbox(
                        value=f"{sd_model_info}\n"
                        f"Images will be saved at "
                        f"{get_generated_imgs_path()}",
                        lines=2,
                        elem_id="std_output",
                        show_label=False,
                    )
                    sd_element.load(logger.read_sd_logs, None, std_output, every=1)
                    sd_status = gr.Textbox(visible=False)
                with gr.Row():
                    stable_diffusion = gr.Button("Generate Image(s)")
                    random_seed = gr.Button("Randomize Seed")
                    random_seed.click(
                        lambda: -1,
                        inputs=[],
                        outputs=[seed],
                        queue=False,
                        show_progress=False,
                    )
                    stop_batch = gr.Button("Stop Batch")
                with gr.Group():
                    with gr.Column(scale=3):
                        sd_json = gr.JSON(
                            value=view_json_file(
                                os.path.join(
                                    get_configs_path(),
                                    "default_sd_config.json",
                                )
                            )
                        )
                    with gr.Column(scale=1):
                        clear_sd_config = gr.ClearButton(
                            value="Clear Config", size="sm", components=sd_json
                        )
                        with gr.Row():
                            save_sd_config = gr.Button(value="Save Config", size="sm")
                            sd_config_name = gr.Textbox(
                                value="Config Name",
                                info="Name of the file this config will be saved to.",
                                interactive=True,
                            )
                        load_sd_config = gr.FileExplorer(
                            label="Load Config",
                            file_count="single",
                            root=(
                                cmd_opts.configs_path
                                if cmd_opts.configs_path
                                else get_configs_path()
                            ),
                            height=75,
                        )
                        load_sd_config.change(
                            fn=load_sd_cfg,
                            inputs=[sd_json, load_sd_config],
                            outputs=[
                                prompt,
                                negative_prompt,
                                sd_init_image,
                                height,
                                width,
                                steps,
                                strength,
                                guidance_scale,
                                seed,
                                batch_count,
                                batch_size,
                                scheduler,
                                base_model_id,
                                custom_weights,
                                custom_vae,
                                precision,
                                device,
                                ondemand,
                                repeatable_seeds,
                                resample_type,
                                cnet_config,
                                embeddings_config,
                                sd_json,
                            ],
                        )
                        save_sd_config.click(
                            fn=save_sd_cfg,
                            inputs=[sd_json, sd_config_name],
                            outputs=[sd_config_name],
                        )

        pull_kwargs = dict(
            fn=pull_sd_configs,
            inputs=[
                prompt,
                negative_prompt,
                sd_init_image,
                height,
                width,
                steps,
                strength,
                guidance_scale,
                seed,
                batch_count,
                batch_size,
                scheduler,
                base_model_id,
                custom_weights,
                custom_vae,
                precision,
                device,
                ondemand,
                repeatable_seeds,
                resample_type,
                cnet_config,
                embeddings_config,
            ],
            outputs=[
                sd_json,
            ],
        )

        status_kwargs = dict(
            fn=lambda bc, bs: status_label("Stable Diffusion", 0, bc, bs),
            inputs=[batch_count, batch_size],
            outputs=sd_status,
        )

        gen_kwargs = dict(
            fn=shark_sd_fn_dict_input,
            inputs=[sd_json],
            outputs=[
                sd_gallery,
                sd_status,
            ],
        )

        prompt_submit = prompt.submit(**status_kwargs).then(**pull_kwargs)
        neg_prompt_submit = negative_prompt.submit(**status_kwargs).then(**pull_kwargs)
        generate_click = (
            stable_diffusion.click(**status_kwargs)
            .then(**pull_kwargs)
            .then(**gen_kwargs)
        )
        stop_batch.click(
            fn=cancel_sd,
            cancels=[prompt_submit, neg_prompt_submit, generate_click],
        )
