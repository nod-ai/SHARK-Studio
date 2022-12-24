import os

os.environ["AMD_ENABLE_LLPC"] = "1"
import gradio as gr
from PIL import Image
from models.stable_diffusion.resources import resource_path, prompt_examples
from models.stable_diffusion.main import stable_diff_inf
from models.stable_diffusion.stable_args import args
from models.stable_diffusion.utils import get_available_devices

nodlogo_loc = resource_path("logos/nod-logo.png")
sdlogo_loc = resource_path("logos/sd-demo-logo.png")


demo_css = """
.gradio-container {background-color: black}
.container {background-color: black !important; padding-top:20px !important; }
#ui_title {padding: 10px !important; }
#top_logo {background-color: transparent; border-radius: 0 !important; border: 0; } 
#demo_title {background-color: black; border-radius: 0 !important; border: 0; padding-top: 50px; padding-bottom: 0px; width: 460px !important;} 

#demo_title_outer  {border-radius: 0; } 
#prompt_box_outer div:first-child  {border-radius: 0 !important}
#prompt_box textarea  {background-color:#1d1d1d !important}
#prompt_examples {margin:0 !important}
#prompt_examples svg {display: none !important;}

.gr-sample-textbox { border-radius: 1rem !important; border-color: rgb(31,41,55) !important; border-width:2px !important; }
#ui_body {background-color: #111111 !important; padding: 10px !important; border-radius: 0.5em !important;}

#img_result+div {display: none !important;}

footer {display: none !important;}
"""


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
                            "anythingv3",
                            "analogdiffusion",
                            "stablediffusion",
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
                            "EulerAncestral",
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
    share=False,
    inbrowser=True,
    server_name="0.0.0.0",
    server_port=8080,
)
