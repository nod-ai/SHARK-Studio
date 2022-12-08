import os
import sys
from pathlib import Path

if "AMD_ENABLE_LLPC" not in os.environ:
    os.environ["AMD_ENABLE_LLPC"] = "1"

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_path, relative_path)


import gradio as gr
from PIL import Image
from apps.stable_diffusion.src import (
    prompt_examples,
    args,
    get_available_devices,
)
from apps.stable_diffusion.scripts import txt2img_inf

nodlogo_loc = resource_path("logos/nod-logo.png")
sdlogo_loc = resource_path("logos/sd-demo-logo.png")


demo_css = resource_path("css/sd_dark_theme.css")


with gr.Blocks(title="Stable Diffusion", css=demo_css) as shark_web:

    with gr.Row(elem_id="ui_title"):
        nod_logo = Image.open(nodlogo_loc)
        logo2 = Image.open(sdlogo_loc)
        with gr.Row():
            with gr.Column(scale=1, elem_id="demo_title_outer"):
                gr.Image(
                    value=nod_logo,
                    show_label=False,
                    interactive=False,
                    elem_id="top_logo",
                ).style(width=150, height=100)
            with gr.Column(scale=5, elem_id="demo_title_outer"):
                gr.Image(
                    value=logo2,
                    show_label=False,
                    interactive=False,
                    elem_id="demo_title",
                ).style(width=150, height=100)

    with gr.Row(elem_id="ui_body"):

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                with gr.Row():
                    with gr.Group():
                        model_id = gr.Dropdown(
                            label="Model ID",
                            value="stabilityai/stable-diffusion-2-1-base",
                            choices=[
                                "Linaqruf/anything-v3.0",
                                "prompthero/openjourney",
                                "wavymulder/Analog-Diffusion",
                                "stabilityai/stable-diffusion-2-1",
                                "stabilityai/stable-diffusion-2-1-base",
                                "CompVis/stable-diffusion-v1-4",
                            ],
                        )
                        custom_model_id = gr.Textbox(
                            placeholder="check here: https://huggingface.co/models eg. runwayml/stable-diffusion-v1-5",
                            value="",
                            label="HuggingFace Model ID",
                        )
                    with gr.Group():
                        ckpt_loc = gr.File(
                            label="Upload checkpoint",
                            file_types=[".ckpt", ".safetensors"],
                        )

                with gr.Group(elem_id="prompt_box_outer"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="cyberpunk forest by Salvador Dali",
                        lines=1,
                        elem_id="prompt_box",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="trees, green",
                        lines=1,
                        elem_id="prompt_box",
                    )
                with gr.Accordion(label="Advance Options", open=False):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            label="Scheduler",
                            value="SharkEulerDiscrete",
                            choices=[
                                "DDIM",
                                "PNDM",
                                "LMSDiscrete",
                                "DPMSolverMultistep",
                                "EulerDiscrete",
                                "EulerAncestralDiscrete",
                                "SharkEulerDiscrete",
                            ],
                        )
                        batch_size = gr.Slider(
                            1, 4, value=1, step=1, label="Number of Images"
                        )
                    with gr.Row():
                        height = gr.Slider(
                            384, 786, value=512, step=8, label="Height"
                        )
                        width = gr.Slider(
                            384, 786, value=512, step=8, label="Width"
                        )
                        precision = gr.Radio(
                            label="Precision",
                            value="fp16",
                            choices=[
                                "fp16",
                                "fp32",
                            ],
                            visible=False,
                        )
                        max_length = gr.Radio(
                            label="Max Length",
                            value=64,
                            choices=[
                                64,
                                77,
                            ],
                            visible=False,
                        )
                    with gr.Row():
                        steps = gr.Slider(
                            1, 100, value=50, step=1, label="Steps"
                        )
                        guidance_scale = gr.Slider(
                            0,
                            50,
                            value=7.5,
                            step=0.1,
                            label="CFG Scale",
                        )
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, label="Seed")
                    available_devices = get_available_devices()
                    device = gr.Dropdown(
                        label="Device",
                        value=available_devices[0],
                        choices=available_devices,
                    )
                with gr.Row():
                    random_seed = gr.Button("Randomize Seed")
                    random_seed.click(
                        None,
                        inputs=[],
                        outputs=[seed],
                        _js="() => Math.floor(Math.random() * 4294967295)",
                    )
                    stable_diffusion = gr.Button("Generate Image")
                with gr.Accordion(label="Prompt Examples!", open=False):
                    ex = gr.Examples(
                        examples=prompt_examples,
                        inputs=prompt,
                        cache_examples=False,
                        elem_id="prompt_examples",
                    )

            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=[2], height="auto")
                    std_output = gr.Textbox(
                        value="Nothing to show.",
                        lines=4,
                        show_label=False,
                    )
                output_dir = args.output_dir if args.output_dir else Path.cwd()
                output_dir = Path(output_dir, "generated_imgs")
                output_loc = gr.Textbox(
                    label="Saving Images at",
                    value=output_dir,
                    interactive=False,
                )

        prompt.submit(
            txt2img_inf,
            inputs=[
                prompt,
                negative_prompt,
                height,
                width,
                steps,
                guidance_scale,
                seed,
                batch_size,
                scheduler,
                model_id,
                custom_model_id,
                ckpt_loc,
                precision,
                device,
                max_length,
            ],
            outputs=[gallery, std_output],
            show_progress=args.progress_bar,
        )
        stable_diffusion.click(
            txt2img_inf,
            inputs=[
                prompt,
                negative_prompt,
                height,
                width,
                steps,
                guidance_scale,
                seed,
                batch_size,
                scheduler,
                model_id,
                custom_model_id,
                ckpt_loc,
                precision,
                device,
                max_length,
            ],
            outputs=[gallery, std_output],
            show_progress=args.progress_bar,
        )

shark_web.queue()
shark_web.launch(
    share=args.share,
    inbrowser=True,
    server_name="0.0.0.0",
    server_port=args.server_port,
)
