import os
import time
import gradio as gr
import PIL
import json
import sys

from math import ceil
from inspect import signature
from PIL import Image
from pathlib import Path
from datetime import datetime as dt
from gradio.components.image_editor import (
    Brush,
    Eraser,
    EditorValue,
)

from apps.shark_studio.api.utils import (
    get_available_devices,
    get_generated_imgs_path,
    get_checkpoints_path,
    get_checkpoints,
)
from apps.shark_studio.api.sd import (
    sd_model_map,
    StableDiffusion,
)
from apps.shark_studio.api.schedulers import (
    scheduler_model_map,
)
from apps.shark_studio.api.controlnet import (
    preprocessor_model_map,
    control_adapter_model_map,
    PreprocessorModel,
)
from apps.shark_studio.modules.img_processing import (
    resampler_list,
    resize_stencil,
)
from apps.shark_studio.web.ui.utils import (
    get_generation_text_info,
    nodlogo_loc,
)
from apps.shark_studio.web.ui.common_events import lora_changed

sd_pipe = None


# NOTE: Each `hf_model_id` should have its own starting configuration.

# model_vmfb_key = ""

def shark_sd_fn(
    prompt: str,
    negative_prompt: str,
    image_dict,
    height: int,
    width: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    seed: str | int,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    base_model_id: str,
    custom_checkpoints: str,
    custom_vae: str,
    precision: str,
    device: str,
    lora_weights: str | list,
    lora_hf_ids: str | list,
    ondemand: bool,
    repeatable_seeds: bool,
    resample_type: str,
    control_mode: str,
    stencils: list,
    images: list,
    preprocessed_hints: list,
    progress=gr.Progress(),
):
    
    # Handling gradio ImageEditor datatypes so we have unified inputs to the SD API
    for i, stencil in enumerate(stencils):
        if images[i] is None and stencil is not None:
            continue
        elif stencil is None and any(img is not None for img in [images[i], preprocessed_hints[i]]):
            images[i] = None
            preprocessed_hints[i] = None
        elif images[i] is not None:
            if isinstance(images[i], dict):
                images[i] = images[i]["composite"]
            images[i] = images[i].convert("RGB")
    
    if isinstance(image_dict, PIL.Image.Image):
        image = image_dict.convert("RGB")
    elif image_dict:
        image = image_dict["image"].convert("RGB")
    else:
        image = None
    if image:
        image, _, _, = resize_stencil(image, width, height)

    device_id = None

    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts

    submit_pipe_kwargs = {
        base_model_id: base_model_id,
        height: height,
        width: width,
        precision: precision,
        device: device,
        extra_model_ids: extra_model_ids,
        embeddings: lora_hf_ids,
        import_ir: cmd_opts.import_ir,
    }
    submit_prep_kwargs = {



    global sd_pipe
    global sd_pipe_kwargs
    
    for key in 

    if sd_pipe is None:
        history[-1][-1] = "Getting the pipeline ready..."
        yield history, ""

        # Initializes the pipeline and retrieves IR based on all
        # parameters that are static in the turbine output format,
        # which is currently MLIR in the torch dialect.

        sd_pipe = SharkStableDiffusionPipeline(
            **submit_pipe_kwargs
        )
        sd_pipe.queue_compile()
    
    for prompt, msg, exec_time in progress.tqdm(
        sd_pipe.generate_images(
            prompt,
            negative_prompt,
            ),
        desc="Generating Image...",
    ):

    return history, ""


def view_json_file(file_obj):
    content = ""
    with open(file_obj.name, "r") as fopen:
        content = fopen.read()
    return content

sd_fn_sig = signature(shark_sd_fn)
max_controlnets = 5
max_loras = 5

def show_loras(k):
    k = int(k)
    return [gr.Dropdown(visible=True)]*k + [gr.Dropdown(visible=False, value="None")]*(max_textboxes-k)

def show_controlnets(k):
    k = int(k)
    return [gr.Row(visible=True)]*k + [gr.Row(visible=False)]*(max_textboxes-k)

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
    resized_img, _, _ = resize_stencil(
        original_img, width, height
    )
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
    width,
    height,
    stencils,
    images,
    preprocessed_hints,
    index,
):
    if model == None:
        stencils[index] = None
        images[index] = None
        preprocessed_hints[index] = None
        return [
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            stencils,
            images,
            preprocessed_hints,
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
            stencils,
            images,
            preprocessed_hints,
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
            gr.Button(visible=True),
            gr.Button(visible=False),
            stencils,
            images,
            preprocessed_hints,
        ]
with gr.Blocks(title="Stable Diffusion") as sd_element:
    # Get a list of arguments needed for the API call, then
    # initialize an empty list that will manage the corresponding
    # gradio values.
    inputs_list = gr.State(signature(shark_sd_fn))
    inputs_args = gr.State([None] * len(inputs_list))
    with gr.Row(elem_id="ui_title"):
            nod_logo = Image.open(nodlogo_loc)
            with gr.Row():
                with gr.Column(scale=1, elem_id="demo_title_outer"):
                    gr.Image(
                        value=nod_logo,
                        show_label=False,
                        interactive=False,
                        elem_id="top_logo",
                        width=150,
                        height=50,
                        show_download_button=False,
                    )
                save_sd_config = gr.Button(label="Save Config", scale=1)
                load_sd_config = gr.FileExplorer("Load Config", scale=1)
                clear_sd_config = gr.ClearButton("Clear Config", scale=1)
    with gr.Column(elem_if="ui_body"):
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                with gr.Group()
                    sd_model_info = (
                        f"Checkpoint Path: {str(get_checkpoint_path())}"
                    )
                    sd_base = gr.Dropdown(
                        label="Base Model",
                        info="Select or enter HF model ID",
                        elem_id="custom_model",
                        value="stabilityai/stable-diffusion-2.1-base",
                        choices=get_base_models(),
                    ) # base_model_id
                    sd_checkpoint = gr.Dropdown(
                        label="Checkpoints (optional)",
                        info="Select or enter HF model ID",
                        elem_id="custom_model",
                        value="None",
                        choices=get_checkpoints(sd_base),
                    ) # 
                    sd_vae_info = (str(get_checkpoints_path("vae"))).replace(
                        "\\", "\n\\"
                    )
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
                    
                with gr.Group(elem_id="prompt_box_outer"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=args.prompts[0],
                        lines=2,
                        elem_id="prompt_box",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=args.negative_prompts[0],
                        lines=2,
                        elem_id="negative_prompt_box",
                    )
                
                with gr.Accordion(label = "Input Image", open=False):
                    # TODO: make this import image prompt info if it exists
                    sd_init_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=300,
                        interactive=True,
                    )
                with gr.Accordion(label="Embeddings options", open=False):
                    sd_lora_info = (
                        str(get_checkpoints_path("loras"))
                    ).replace("\\", "\n\\")
                    num_loras = gr.Slider(1, max_loras, value=1, step=1, label="LoRA Count")
                    loras = []
                    for i in range(max_loras):
                        lora_opt = gr.Dropdown(
                            allow_custom_value=False,
                            label=f"Standalone LoRA Weights",
                            info=sd_lora_info,
                            elem_id="lora_weights",
                            value="None",
                            choices=["None"] + get_custom_model_files("lora"),
                        )
                with gr.Accordion(label="Advanced Options", open=True):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            elem_id="scheduler",
                            label="Scheduler",
                            value="EulerDiscrete",
                            choices=scheduler_list,
                            allow_custom_value=False,
                        )
                    with gr.Row():
                        height = gr.Slider(
                            384, 768, value=cmd_opts.height, step=8, label="Height"
                        )
                        width = gr.Slider(
                            384, 768, value=cmd_opts.width, step=8, label="Width"
                        )
                    with gr.Row():
                        with gr.Column(scale=3):
                            steps = gr.Slider(
                                1, 100, value=args.steps, step=1, label="Steps"
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
                    with gr.Row():
                        with gr.Column(scale=3):
                            guidance_scale = gr.Slider(
                                0,
                                50,
                                value=cmd_opts.guidance_scale,
                                step=0.1,
                                label="CFG Scale",
                            )
                        with gr.Column(scale=3):
                            batch_count = gr.Slider(
                                1,
                                100,
                                value=cmd_opts.batch_count,
                                step=1,
                                label="Batch Count",
                                interactive=True,
                            )
                        repeatable_seeds = gr.Checkbox(
                            cmd_opts.repeatable_seeds,
                            label="Repeatable Seeds",
                        )
                    with gr.Row():
                        batch_size = gr.Slider(
                            1,
                            4,
                            value=cmd_opts.batch_size,
                            step=1,
                            label="Batch Size",
                            interactive=True,
                            visible=True,
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
                        value=get_available_devices[0],
                        choices=get_available_devices,
                        allow_custom_value=False,
                    )
                with gr.Accordion(label="Controlnet Options", open=False):
                    sd_cnet_info = (
                        str(get_checkpoints_path("controlnet"))
                    ).replace("\\", "\n\\")
                    num_cnets = gr.Slider(1, max_controlnets, value=1, step=1, label="Controlnet Count")
                    cnet_rows = []
                    stencils = []
                    images = []
                    preprocessed_hints = []
                    for i in range(max_controlnets):
                        with gr.Row as cnet_row:
                            with gr.Column():
                                cnet_gen = gr.Button(
                                    value="Preprocess controlnet input",
                                )
                                cnet_processor = gr.Dropdown(
                                    allow_custom_value=True,
                                    label=f"Controlnet Preprocessor",
                                    info=sd_cnet_info,
                                    elem_id="lora_weights",
                                    value="None",
                                    choices=["None"] + controlnet_list + get_custom_model_files("controlnet"),
                                )
                                cnet_adapter = gr.Dropdown(
                                    allow_custom_value=True,
                                    label=f"Controlnet Adapter",
                                    info=sd_cnet_info,
                                    elem_id="lora_weights",
                                    value="None",
                                    choices=["None"] + controlnet_list + get_custom_model_files("controlnet"),
                                )
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
                                )
                            cnet_input = gr.ImageEditor(
                                visible=True,
                                image_mode="RGB",
                                interactive=True,
                                show_label=True,
                                label="Input Image",
                                type="pil",
                            )
                            cnet_output = gr.Image(
                                value=None,
                                visible=True,
                                label="Preprocessed Hint",
                                interactive=True,
                                show_label=True
                            )
                            use_input_img.click(
                                import_original,
                                [sd_init_image, canvas_width, canvas_height],
                                [cnet_image],
                            )

                            cnet_model.change(
                                fn=update_cn_input,
                                inputs=[
                                    cnet_model,
                                    canvas_width,
                                    canvas_height,
                                    stencils,
                                    images,
                                    preprocessed_hints,
                                ],
                                outputs=[
                                    cnet_input,
                                    cnet_output,
                                    canvas_width,
                                    canvas_height,
                                    make_canvas,
                                    use_input_img,
                                    stencils,
                                    images,
                                    preprocessed_hints,
                                ],
                            )
                            make_canvas.click(
                                create_canvas,
                                [canvas_width, canvas_height],
                                [
                                    cnet_image,
                                ],
                            )
                            gr.on(
                                triggers=[cnet_gen.click],
                                fn=cnet_preview,
                                inputs=[
                                    cnet_model,
                                    cnet_input,
                                    stencils,
                                    images,
                                    preprocessed_hints,
                                ],
                                outputs=[
                                    cnet_output,
                                    stencils,
                                    images,
                                    preprocessed_hints,
                                ],
                            )
                            cnet_rows.append(cnet_row)

                        num_cnets.change(show_controlnets, num_cnets, cnet_rows)
            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    img2img_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                        columns=2,
                        object_fit="contain",
                    )
                    std_output = gr.Textbox(
                        value=f"{i2i_model_info}\n"
                        f"Images will be saved at "
                        f"{get_generated_imgs_path()}",
                        lines=2,
                        elem_id="std_output",
                        show_label=False,
                    )
                    img2img_status = gr.Textbox(visible=False)
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
                sd_checkpoint,
                sd_custom_vae,
                precision,
                device,
                lora_weights,
                lora_hf_id,
                ondemand,
                repeatable_seeds,
                resample_type,
                control_mode,
                stencils,
                images,
                preprocessed_hints,
            ],
            outputs=[
                sd_gallery,
                std_output,
                sd_status,
                stencils,
                images,
            ],
            show_progress="minimal" if cmd_opts.progress_bar else "none",
        )

        status_kwargs = dict(
            fn=lambda bc, bs: status_label("Image-to-Image", 0, bc, bs),
            inputs=[batch_count, batch_size],
            outputs=img2img_status,
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

        lora_weights.change(
            fn=lora_changed,
            inputs=[lora_weights],
            outputs=[lora_tags],
            queue=True,
        )
