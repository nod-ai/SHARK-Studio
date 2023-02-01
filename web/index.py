import os
import sys
from pathlib import Path

if "AMD_ENABLE_LLPC" not in os.environ:
    os.environ["AMD_ENABLE_LLPC"] = "1"

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"

import gradio as gr
from PIL import Image
from models.stable_diffusion.resources import resource_path, prompt_examples
from models.stable_diffusion.main import stable_diff_inf
from models.stable_diffusion.stable_args import args
from models.stable_diffusion.utils import get_available_devices

nodlogo_loc = resource_path("logos/nod-logo.png")
sdlogo_loc = resource_path("logos/sd-demo-logo.png")


demo_css = Path(__file__).parent.joinpath("demo.css").resolve()


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
                with gr.Row():
                    variant = gr.Dropdown(
                        label="Model Variant",
                        value="stablediffusion",
                        choices=[
                            "stablediffusion",
                            "anythingv3",
                            "analogdiffusion",
                            "openjourney",
                            "dreamlike",
                        ],
                    )
                    scheduler_key = gr.Dropdown(
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
                with gr.Row():
                    steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
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
                    device_key = gr.Dropdown(
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
                with gr.Accordion(label="Prompt Examples!"):
                    ex = gr.Examples(
                        examples=prompt_examples,
                        inputs=prompt,
                        cache_examples=False,
                        elem_id="prompt_examples",
                    )

            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    generated_img = gr.Image(
                        type="pil", interactive=False
                    ).style(height=512)
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
            stable_diff_inf,
            inputs=[
                prompt,
                negative_prompt,
                steps,
                guidance_scale,
                seed,
                scheduler_key,
                variant,
                device_key,
            ],
            outputs=[generated_img, std_output],
            show_progress=args.progress_bar,
        )
        stable_diffusion.click(
            stable_diff_inf,
            inputs=[
                prompt,
                negative_prompt,
                steps,
                guidance_scale,
                seed,
                scheduler_key,
                variant,
                device_key,
            ],
            outputs=[generated_img, std_output],
            show_progress=args.progress_bar,
        )

shark_web.queue()
shark_web.launch(
    share=args.share,
    inbrowser=True,
    server_name="0.0.0.0",
    server_port=args.server_port,
)
