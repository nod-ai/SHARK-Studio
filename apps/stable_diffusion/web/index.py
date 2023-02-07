import os
import sys
from pathlib import Path
import glob

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
                    ckpt_path = (
                        Path(args.ckpt_dir)
                        if args.ckpt_dir
                        else Path(Path.cwd(), "models")
                    )
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    types = (
                        "*.ckpt",
                        "*.safetensors",
                    )  # the tuple of file types
                    ckpt_files = ["None"]
                    for extn in types:
                        files = glob.glob(os.path.join(ckpt_path, extn))
                        ckpt_files.extend(files)
                    custom_model = gr.Dropdown(
                        label=f"Models (Custom Model path: {ckpt_path})",
                        value="None",
                        choices=ckpt_files
                        + [
                            "Linaqruf/anything-v3.0",
                            "prompthero/openjourney",
                            "wavymulder/Analog-Diffusion",
                            "stabilityai/stable-diffusion-2-1",
                            "stabilityai/stable-diffusion-2-1-base",
                            "CompVis/stable-diffusion-v1-4",
                        ],
                    )
                    hf_model_id = gr.Textbox(
                        placeholder="Select 'None' in the Models dropdown on the left and enter model ID here e.g: SG161222/Realistic_Vision_V1.3",
                        value="",
                        label="HuggingFace Model ID",
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
                with gr.Accordion(label="Advanced Options", open=False):
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
                        with gr.Group():
                            save_metadata_to_png = gr.Checkbox(
                                label="Save prompt information to PNG",
                                value=True,
                                interactive=True,
                            )
                            save_metadata_to_json = gr.Checkbox(
                                label="Save prompt information to JSON file",
                                value=False,
                                interactive=True,
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
                        batch_count = gr.Slider(
                            1,
                            10,
                            value=1,
                            step=1,
                            label="Batch Count",
                            interactive=True,
                        )
                        batch_size = gr.Slider(
                            1,
                            4,
                            value=1,
                            step=1,
                            label="Batch Size",
                            interactive=True,
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
        kwargs = dict(
            fn=txt2img_inf,
            inputs=[
                prompt,
                negative_prompt,
                height,
                width,
                steps,
                guidance_scale,
                seed,
                batch_count,
                batch_size,
                scheduler,
                custom_model,
                hf_model_id,
                precision,
                device,
                max_length,
                save_metadata_to_json,
                save_metadata_to_png,
            ],
            outputs=[gallery, std_output],
            show_progress=args.progress_bar,
        )

        prompt.submit(**kwargs)
        stable_diffusion.click(**kwargs)

shark_web.queue()
shark_web.launch(
    share=args.share,
    inbrowser=True,
    server_name="0.0.0.0",
    server_port=args.server_port,
)
