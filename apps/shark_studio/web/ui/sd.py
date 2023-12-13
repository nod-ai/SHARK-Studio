import os
import json
import gradio as gr
import numpy as np

from math import ceil
from inspect import signature
from PIL import Image
from pathlib import Path
from datetime import datetime as dt
from gradio.components.image_editor import (
    Brush,
    EditorValue,
)

from apps.shark_studio.api.utils import (
    get_available_devices,
)
from apps.shark_studio.web.utils.file_utils import (
    get_generated_imgs_path,
    get_checkpoints_path,
    get_checkpoints,
    get_configs_path,
)
from apps.shark_studio.api.sd import (
    sd_model_map,
    shark_sd_fn,
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
)
from apps.shark_studio.web.utils.state import (
    status_label,
)
from apps.shark_studio.web.ui.common_events import lora_changed


def view_json_file(file_obj):
    content = ""
    with open(file_obj.name, "r") as fopen:
        content = fopen.read()
    return content


def submit_to_cnet_config(stencil: str, preprocessed_hint: str, cnet_strength: int, curr_config: dict):
    if any(i in [None, ""] for i in [stencil, preprocessed_hint]):
        return gr.update()
    if curr_config is not None:
        if "controlnets" in curr_config:           
            curr_config["controlnets"]["model"].append(stencil)
            curr_config["controlnets"]["hint"].append(preprocessed_hint)
            curr_config["controlnets"]["strength"].append(cnet_strength)
            return curr_config

    cnet_map = {}
    cnet_map["controlnets"] = {
        "model": [stencil],
        "hint": [preprocessed_hint],
        "strength": [cnet_strength],
    }
    return cnet_map


def update_embeddings_json(embedding, curr_config: dict):
    if curr_config is not None:
        if "embeddings" in curr_config:
            curr_config["embeddings"].append(embedding)
            return curr_config

    config = {"embeddings": [embedding]}

    return config


def submit_to_main_config(input_cfg: dict, main_cfg: dict):
    if main_cfg in [None, ""]:
        # only time main_cfg should be a string is empty case.
        return input_cfg

    for base_key in input_cfg:
        main_cfg[base_key] = input_cfg[base_key]
    return main_cfg


def save_sd_cfg(config: dict, save_name: str):
    if os.path.exists(save_name):
        filepath=save_name
    elif cmd_opts.configs_path:
        filepath=os.path.join(cmd_opts.configs_path, save_name)
    else:
        filepath=os.path.join(get_configs_path(), save_name)
    if ".json" not in filepath:
        filepath += ".json"
    with open(filepath, mode="w") as f:
        f.write(json.dumps(config))
    return("...")

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
        "layers": [data],
        "composite": None,
    }
    return EditorValue(img_dict)


def import_original(original_img, width, height):
    if original_img is None:
        resized_img = create_canvas(width, height)
        return gr.ImageEditor(
            value=resized_img,
            crop_size=(width, height),
        )
    else:
        resized_img, _, _ = resize_stencil(original_img, width, height)
        img_dict = {
            "background": resized_img,
            "layers": [resized_img],
            "composite": None,
        }
        return gr.ImageEditor(
            value=EditorValue(img_dict),
            crop_size=(width, height),
        )


def update_cn_input(
    model,
    stencil,
    preprocessed_hint,
):
    print("update_cn_input")
    if model == None:
        stencil = None
        preprocessed_hint = None
        return [
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            stencil,
            preprocessed_hint,
        ]
    elif model == "scribble":
        return [
            gr.ImageEditor(
                visible=True,
                interactive=True,
                show_label=False,
                image_mode="RGB",
                type="pil",
                brush=Brush(
                    colors=["#000000"],
                    color_mode="fixed",
                    default_size=5,
                ),
            ),
            gr.Image(
                visible=True,
                show_label=False,
                interactive=True,
                show_download_button=False,
            ),
            gr.Slider(visible=True, label="Canvas Width"),
            gr.Slider(visible=True, label="Canvas Height"),
            gr.Button(visible=True),
            gr.Button(visible=False),
            stencil,
            preprocessed_hint,
        ]
    else:
        return [
            gr.ImageEditor(
                visible=True,
                interactive=True,
                show_label=False,
                image_mode="RGB",
                type="pil",
            ),
            gr.Image(
                visible=True,
                show_label=False,
                interactive=True,
                show_download_button=False,
            ),
            gr.Slider(visible=True, label="Canvas Width"),
            gr.Slider(visible=True, label="Canvas Height"),
            gr.Button(visible=False),
            gr.Button(visible=True),
            stencil,
            preprocessed_hint,
        ]


sd_fn_inputs = []
sd_fn_sig = signature(shark_sd_fn).replace()
for i in sd_fn_sig.parameters:
    sd_fn_inputs.append(i)

with gr.Blocks(title="Stable Diffusion") as sd_element:
    # Get a list of arguments needed for the API call, then
    # initialize an empty list that will manage the corresponding
    # gradio values.
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
            with gr.Column(scale=1, min_width=600):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        sd_model_info = (
                            f"Checkpoint Path: {str(get_checkpoints_path())}"
                        )
                        sd_base = gr.Dropdown(
                            label="Base Model",
                            info="Select or enter HF model ID",
                            elem_id="custom_model",
                            value="stabilityai/stable-diffusion-2-1-base",
                            choices=sd_model_map.keys(),
                        )  # base_model_id
                        sd_custom_weights = gr.Dropdown(
                            label="Custom Weights",
                            info="Select or enter HF model ID",
                            elem_id="custom_model",
                            value="None",
                            allow_custom_value=True,
                            choices=get_checkpoints(sd_base),
                        )  #
                    with gr.Column(scale=2):
                        sd_vae_info = (
                            str(get_checkpoints_path("vae"))
                        ).replace("\\", "\n\\")
                        sd_vae_info = f"VAE Path: {sd_vae_info}"
                        sd_custom_vae = gr.Dropdown(
                            label=f"Custom VAE Models",
                            info=sd_vae_info,
                            elem_id="custom_model",
                            value=os.path.basename(cmd_opts.custom_vae)
                            if cmd_opts.custom_vae
                            else "None",
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
                        value=cmd_opts.prompts[0],
                        lines=2,
                        elem_id="prompt_box",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=cmd_opts.negative_prompts[0],
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
                with gr.Accordion(
                    label="Embeddings options", open=True, render=True
                ):
                    sd_lora_info = (
                        str(get_checkpoints_path("loras"))
                    ).replace("\\", "\n\\")                 
                    with gr.Column(scale=2):
                        lora_opt = gr.Dropdown(
                            allow_custom_value=True,
                            label=f"Standalone LoRA Weights",
                            info=sd_lora_info,
                            elem_id="lora_weights",
                            value="None",
                            choices=["None"] + get_checkpoints("lora"),
                        )
                        lora_tags = gr.HTML(
                            value="<div><i>No LoRA selected</i></div>",
                            elem_classes="lora-tags",
                        )
                    with gr.Column(scale=1):
                        embeddings_config = gr.JSON()
                        submit_embeddings = gr.Button("Submit to Main Config", size="sm")       
                    gr.on(
                        triggers=[lora_opt.change],
                        fn=lora_changed,
                        inputs=[lora_opt],
                        outputs=[lora_tags],
                        queue=True,
                    )
                    gr.on(
                        triggers=[lora_opt.change],
                        fn=update_embeddings_json,
                        inputs=[lora_opt, embeddings_config],
                        outputs=[embeddings_config],
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
                        value=get_available_devices()[0],
                        choices=get_available_devices(),
                        allow_custom_value=False,
                    )
                with gr.Accordion(
                    label="Controlnet Options", open=False, render=True
                ):
                    with gr.Column():
                        cnet_config = gr.JSON()
                        submit_cnet = gr.Button("Submit to Main Config", size="sm")
                    with gr.Column():
                        sd_cnet_info = (
                            str(get_checkpoints_path("controlnet"))
                        ).replace("\\", "\n\\")
                        stencil = gr.State("")
                        preprocessed_hint = gr.State("")
                    with gr.Row():
                        control_mode = gr.Radio(
                            choices=["Prompt", "Balanced", "Controlnet"],
                            value="Balanced",
                            label="Control Mode",
                        )

                    with gr.Row(visible=True) as cnet_row:
                        with gr.Column():
                            cnet_gen = gr.Button(
                                value="Preprocess controlnet input",
                            )
                            cnet_model = gr.Dropdown(
                                allow_custom_value=True,
                                label=f"Controlnet Model",
                                info=sd_cnet_info,
                                elem_id="lora_weights",
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
                                    step=1,
                                    visible=False,
                                )
                                canvas_height = gr.Slider(
                                    label="Canvas Height",
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=1,
                                    visible=False,
                                )
                            make_canvas = gr.Button(
                                value="Make Canvas!",
                                visible=False,
                            )
                            use_input_img = gr.Button(
                                value="Use Original Image",
                                visible=False,
                                size="sm",
                            )
                        cnet_input = gr.ImageEditor(
                            visible=True,
                            image_mode="RGB",
                            interactive=True,
                            show_label=True,
                            label="Input Image",
                            type="pil",
                        )
                        with gr.Column():
                            cnet_output = gr.Image(
                                value=None,
                                visible=True,
                                label="Preprocessed Hint",
                                interactive=True,
                                show_label=True,
                            )
                            use_result = gr.Button(
                                "Submit",
                                size="sm",
                            )
                        use_input_img.click(
                            import_original,
                            [sd_init_image, canvas_width, canvas_height],
                            [cnet_input],
                        )
                        cnet_model.change(
                            fn=update_cn_input,
                            inputs=[
                                cnet_model,
                                stencil,
                                preprocessed_hint,
                            ],
                            outputs=[
                                cnet_input,
                                cnet_output,
                                canvas_width,
                                canvas_height,
                                make_canvas,
                                use_input_img,
                                stencil,
                                preprocessed_hint,
                            ],
                        )
                        make_canvas.click(
                            create_canvas,
                            [canvas_width, canvas_height],
                            [
                                cnet_input,
                            ],
                        )
                        gr.on(
                            triggers=[cnet_gen.click],
                            fn=cnet_preview,
                            inputs=[
                                cnet_model,
                                cnet_input,
                                stencil,
                                preprocessed_hint,
                            ],
                            outputs=[
                                cnet_output,
                                stencil,
                                preprocessed_hint,
                            ],
                        )
                        use_result.click(
                            fn=submit_to_cnet_config,
                            inputs=[
                                stencil,
                                preprocessed_hint,
                                cnet_strength,
                                cnet_config,
                            ],
                            outputs=[
                                cnet_config,
                            ]
                        )
            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    sd_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                        columns=2,
                        object_fit="contain",
                    )
                    std_output = gr.Textbox(
                        value=f"{sd_model_info}\n"
                        f"Images will be saved at "
                        f"{get_generated_imgs_path()}",
                        lines=2,
                        elem_id="std_output",
                        show_label=False,
                    )
                    sd_status = gr.Textbox(visible=False)
                with gr.Row():
                    stable_diffusion = gr.Button("Generate Image(s)")
                    random_seed = gr.Button("Randomize Seed")
                    random_seed.click(
                        lambda: -1,
                        inputs=[],
                        outputs=[seed],
                        queue=False,
                    )
                    stop_batch = gr.Button("Stop Batch")
                with gr.Group():
                    sd_json = gr.JSON()
                    with gr.Row():
                        clear_sd_config = gr.ClearButton(
                            value="Clear Config", size="sm"
                        )
                        save_sd_config = gr.Button(
                            value="Save Config", size="sm"
                        )
                        sd_config_name = gr.Textbox(
                            value="Config Name",
                            info="Name of the file this config will be saved to.",
                            interactive=True,
                        )
                        load_sd_config = gr.FileExplorer(
                            label="Load Config",
                            root=cmd_opts.configs_path if cmd_opts.configs_path else get_configs_path(),
                            height=75,
                        )
                        save_sd_config.click(
                            fn=save_sd_cfg,
                            inputs=[sd_json, sd_config_name],
                            outputs=[sd_config_name],
                        )

        kwargs = dict(
            fn=shark_sd_fn,
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
                sd_base,
                sd_custom_weights,
                sd_custom_vae,
                precision,
                device,
                ondemand,
                repeatable_seeds,
                resample_type,
                control_mode,
                sd_json,
            ],
            outputs=[
                sd_gallery,
                std_output,
                sd_status,
            ],
            show_progress="minimal",
        )

        status_kwargs = dict(
            fn=lambda bc, bs: status_label("Stable Diffusion", 0, bc, bs),
            inputs=[batch_count, batch_size],
            outputs=sd_status,
        )

        prompt_submit = prompt.submit(**status_kwargs).then(**kwargs)
        neg_prompt_submit = negative_prompt.submit(**status_kwargs).then(
            **kwargs
        )
        generate_click = stable_diffusion.click(**status_kwargs).then(**kwargs)
        stop_batch.click(
            fn=cancel_sd,
            cancels=[prompt_submit, neg_prompt_submit, generate_click],
        )
        gr.on(
            triggers=[submit_cnet.click],
            fn=submit_to_main_config,
            inputs=[cnet_config, sd_json],
            outputs=[sd_json],
        )
        gr.on(
            triggers=[submit_embeddings.click],
            fn=submit_to_main_config,
            inputs=[embeddings_config, sd_json],
            outputs=[sd_json],
        )
