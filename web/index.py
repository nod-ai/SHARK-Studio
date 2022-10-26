# from models.resnet50 import resnet_inf
# from models.albert_maskfill import albert_maskfill_inf
from models.stable_diffusion.main import stable_diff_inf
from models.esrgan import esrgan_inf

# from models.diffusion.v_diffusion import vdiff_inf
import gradio as gr
from PIL import Image
import json
import os


def debug_event(debug):
    return gr.Textbox.update(visible=debug)


prompt_examples = []
prompt_loc = "./prompts.json"
if os.path.exists(prompt_loc):
    with open("./prompts.json", encoding="utf-8") as fopen:
        prompt_examples = json.load(fopen)


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

with gr.Blocks(css=demo_css) as shark_web:

    with gr.Row(elem_id="ui_title"):
        nod_logo = Image.open("./logos/nod-logo.png")
        logo2 = Image.open("./logos/sd-demo-logo.png")
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

    with gr.Tabs():
        with gr.TabItem("Stable-Diffusion"):
            prompt = (
                scheduler
            ) = (
                iters_count
            ) = (
                batch_size
            ) = (
                steps
            ) = (
                guidance
            ) = (
                height
            ) = (
                width
            ) = (
                seed
            ) = (
                precision
            ) = (
                device
            ) = (
                cache
            ) = (
                iree_vulkan_target_triple
            ) = (
                live_preview
            ) = (
                debug
            ) = save_img = stable_diffusion = generated_img = std_output = None
            # load prompts.

            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    with gr.Group(elem_id="prompt_box_outer"):
                        prompt = gr.Textbox(
                            label="Prompt",
                            value="A photograph of an astronaut riding a horse",
                            lines=1,
                            elem_id="prompt_box",
                        )
                    with gr.Group():
                        ex = gr.Examples(
                            label="Examples",
                            examples=prompt_examples,
                            inputs=prompt,
                            cache_examples=False,
                            elem_id="prompt_examples",
                        )
                    with gr.Row():
                        steps = gr.Slider(
                            1, 100, value=50, step=1, label="Steps"
                        )
                        guidance = gr.Slider(
                            0,
                            50,
                            value=7.5,
                            step=0.1,
                            label="Guidance Scale",
                            interactive=False,
                        )
                    with gr.Row():
                        height = gr.Slider(
                            384,
                            768,
                            value=512,
                            step=64,
                            label="Height",
                            interactive=False,
                        )
                        width = gr.Slider(
                            384,
                            768,
                            value=512,
                            step=64,
                            label="Width",
                            interactive=False,
                        )
                    with gr.Row():
                        precision = gr.Radio(
                            label="Precision",
                            value="fp16",
                            choices=["fp16", "fp32"],
                        )
                        seed = gr.Textbox(
                            value="42", max_lines=1, label="Seed"
                        )
                    with gr.Row():
                        cache = gr.Checkbox(label="Cache", value=True)
                        debug = gr.Checkbox(label="DEBUG", value=False)
                        save_img = gr.Checkbox(label="Save Image", value=False)
                        live_preview = gr.Checkbox(
                            label="Live Preview", value=False
                        )
                        # Hidden Items.
                        scheduler = gr.Radio(
                            label="Scheduler",
                            value="LMS",
                            choices=["PNDM", "LMS", "DDIM"],
                            interactive=False,
                            visible=False,
                        )
                        device = gr.Radio(
                            label="Device",
                            value="vulkan",
                            choices=["cpu", "cuda", "vulkan"],
                            interactive=False,
                            visible=False,
                            elem_id="ugly_line",
                        )
                        iters_count = gr.Slider(
                            1,
                            24,
                            value=1,
                            step=1,
                            label="Iteration Count",
                            visible=False,
                        )
                        batch_size = gr.Slider(
                            1,
                            4,
                            value=1,
                            step=1,
                            label="Batch Size",
                            visible=False,
                        )
                        iree_vulkan_target_triple = gr.Textbox(
                            value="",
                            max_lines=1,
                            label="IREE VULKAN TARGET TRIPLE",
                            visible=False,
                            elem_id="ugly_line",
                        )
                    stable_diffusion = gr.Button("Generate Image")
                with gr.Column(scale=1, min_width=600):
                    generated_img = gr.Image(
                        type="pil", elem_id="img_result", interactive=False
                    ).style(height=768)
                    std_output = gr.Textbox(
                        label="Std Output",
                        value="Nothing.",
                        lines=5,
                        visible=False,
                        elem_id="ugly_line",
                    )

            debug.change(
                debug_event,
                inputs=[debug],
                outputs=[std_output],
                show_progress=False,
            )

            stable_diffusion.click(
                stable_diff_inf,
                inputs=[
                    prompt,
                    scheduler,
                    iters_count,
                    batch_size,
                    steps,
                    guidance,
                    height,
                    width,
                    seed,
                    precision,
                    device,
                    cache,
                    iree_vulkan_target_triple,
                    live_preview,
                    save_img,
                ],
                outputs=[generated_img, std_output],
            )
        with gr.TabItem("Upscaler"):
            image = device = debug = esrgan = output = std_output = None
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    image = gr.Image(label="Image")
                    device = gr.Radio(
                        label="Device",
                        value="cpu",
                        choices=["cpu", "cuda", "vulkan"],
                    )
                    debug = gr.Checkbox(label="DEBUG", value=False)
                    esrgan = gr.Button("Upscale Image").style(full_width=True)
                with gr.Column(scale=1, min_width=600):
                    generated_img = gr.Image(type="pil", shape=(100, 100))
                    std_output = gr.Textbox(
                        label="Std Output",
                        value="Nothing.",
                        lines=10,
                        visible=False,
                    )
            debug.change(
                debug_event,
                inputs=[debug],
                outputs=[std_output],
                show_progress=False,
            )
            esrgan.click(
                esrgan_inf,
                inputs=[image, device],
                outputs=[generated_img, std_output],
            )

shark_web.queue()
shark_web.launch(
    server_name="0.0.0.0", server_port=8081, share=True, enable_queue=True
)
