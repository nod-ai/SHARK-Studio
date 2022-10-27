from models.resnet50 import resnet_inf
from models.albert_maskfill import albert_maskfill_inf
from models.stable_diffusion.main import stable_diff_inf

#  from models.diffusion.v_diffusion import vdiff_inf
import gradio as gr
from PIL import Image


def debug_event(debug):
    return gr.Textbox.update(visible=debug)


with gr.Blocks() as shark_web:

    with gr.Row():
        with gr.Group():
            with gr.Column(scale=1):
                img = Image.open("./logos/Nod_logo.jpg")
                gr.Image(value=img, show_label=False, interactive=False).style(
                    height=300, width=400
                )
            with gr.Column(scale=9):
                gr.Label(value="Shark Models Demo.")

    with gr.Tabs():
        with gr.TabItem("ResNet50"):
            image = device = debug = resnet = output = std_output = None
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    image = gr.Image(label="Image")
                    device = gr.Radio(
                        label="Device",
                        value="cpu",
                        choices=["cpu", "cuda", "vulkan"],
                    )
                    debug = gr.Checkbox(label="DEBUG", value=False)
                    resnet = gr.Button("Recognize Image").style(
                        full_width=True
                    )
                with gr.Column(scale=1, min_width=600):
                    output = gr.Label(label="Output")
                    std_output = gr.Textbox(
                        label="Std Output",
                        value="Nothing to show.",
                        visible=False,
                    )
            debug.change(
                debug_event,
                inputs=[debug],
                outputs=[std_output],
                show_progress=False,
            )
            resnet.click(
                resnet_inf,
                inputs=[image, device],
                outputs=[output, std_output],
            )

        with gr.TabItem("Albert MaskFill"):
            masked_text = (
                device
            ) = debug = albert_mask = decoded_res = std_output = None
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    masked_text = gr.Textbox(
                        label="Masked Text",
                        placeholder="Give me a sentence with [MASK] to fill",
                    )
                    device = gr.Radio(
                        label="Device",
                        value="cpu",
                        choices=["cpu", "cuda", "vulkan"],
                    )
                    debug = gr.Checkbox(label="DEBUG", value=False)
                    albert_mask = gr.Button("Decode Mask")
                with gr.Column(scale=1, min_width=600):
                    decoded_res = gr.Label(label="Decoded Results")
                    std_output = gr.Textbox(
                        label="Std Output",
                        value="Nothing to show.",
                        visible=False,
                    )
            debug.change(
                debug_event,
                inputs=[debug],
                outputs=[std_output],
                show_progress=False,
            )
            albert_mask.click(
                albert_maskfill_inf,
                inputs=[masked_text, device],
                outputs=[decoded_res, std_output],
            )

        #  with gr.TabItem("V-Diffusion"):
        #      prompt = sample_count = batch_size = iters = device = v_diffusion = generated_img = None
        #      with gr.Row():
        #          with gr.Column(scale=1, min_width=600):
        #              prompt = gr.Textbox(
        #                  label="Prompt", value="New York City, oil on canvas:5"
        #              )
        #              sample_count = gr.Number(label="Sample Count", value=1)
        #              batch_size = gr.Number(label="Batch Size", value=1)
        #              iters = gr.Number(label="Steps", value=2)
        #              device = gr.Textbox(label="Device", value="gpu")
        #              v_diffusion = gr.Button("Generate image from prompt")
        #          with gr.Column(scale=1, min_width=600):
        #              generated_img = gr.Image(type="pil", shape=(100, 100))
        #              std_output = gr.Textbox(label="Std Output", value="Nothing.")
        #      v_diffusion.click(
        #          vdiff_inf,
        #          inputs=[prompt, sample_count, batch_size, iters, device],
        #          outputs=[generated_img, std_output]
        #      )

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
            ) = debug = stable_diffusion = generated_img = std_output = None
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="a photograph of an astronaut riding a horse",
                        lines=2,
                    )
                    scheduler = gr.Radio(
                        label="Scheduler",
                        value="LMS",
                        choices=["PNDM", "LMS", "DDIM"],
                        visible=False,
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
                    steps = gr.Slider(1, 100, value=20, step=1, label="Steps")
                    guidance = gr.Slider(
                        0, 50, value=7.5, step=0.1, label="Guidance Scale"
                    )
                    height = gr.Slider(
                        384, 768, value=512, step=64, label="Height"
                    )
                    width = gr.Slider(
                        384, 768, value=512, step=64, label="Width"
                    )
                    seed = gr.Textbox(value="42", max_lines=1, label="Seed")
                    precision = gr.Radio(
                        label="Precision",
                        value="fp32",
                        choices=["fp16", "fp32"],
                    )
                    device = gr.Radio(
                        label="Device",
                        value="vulkan",
                        choices=["cpu", "cuda", "vulkan"],
                    )
                    debug = gr.Checkbox(label="DEBUG", value=False)
                    stable_diffusion = gr.Button("Generate image from prompt")
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
                ],
                outputs=[generated_img, std_output],
            )

shark_web.launch(share=True, server_port=8080, enable_queue=True)
