from models.resnet50 import resnet_inf
from models.albert_maskfill import albert_maskfill_inf
from models.stable_diffusion import stable_diff_inf

#  from models.diffusion.v_diffusion import vdiff_inf
import gradio as gr
from PIL import Image


def debug_event(debug):
    return gr.Textbox.update(visible=debug)


with gr.Blocks() as shark_web:

    with gr.Row():
        with gr.Group():
            with gr.Column(scale=1):
                img = Image.open("./Nod_logo.jpg")
                gr.Image(value=img, show_label=False, interactive=False).style(
                    height=70, width=70
                )
            with gr.Column(scale=9):
                gr.Label(value="Shark Models Demo.")

    with gr.Tabs():
        with gr.TabItem("ResNet50"):
            image = device = debug = resnet = output = std_output = None
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    image = gr.Image(label="Image")
                    device = gr.Textbox(label="Device", value="cpu")
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
                    device = gr.Textbox(label="Device", value="cpu")
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
                iters
            ) = (
                device
            ) = debug = stable_diffusion = generated_img = std_output = None
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="a photograph of an astronaut riding a horse",
                    )
                    iters = gr.Number(label="Steps", value=2)
                    device = gr.Textbox(label="Device", value="vulkan")
                    debug = gr.Checkbox(label="DEBUG", value=False)
                    stable_diffusion = gr.Button("Generate image from prompt")
                with gr.Column(scale=1, min_width=600):
                    generated_img = gr.Image(type="pil", shape=(100, 100))
                    std_output = gr.Textbox(
                        label="Std Output", value="Nothing.", visible=False
                    )
            debug.change(
                debug_event,
                inputs=[debug],
                outputs=[std_output],
                show_progress=False,
            )
            stable_diffusion.click(
                stable_diff_inf,
                inputs=[prompt, iters, device],
                outputs=[generated_img, std_output],
            )

shark_web.launch(share=True, server_port=8080, enable_queue=True)
