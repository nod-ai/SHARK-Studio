from models.resnet50 import resnet_inf
from models.albert_maskfill import albert_maskfill_inf
from models.diffusion.v_diffusion import vdiff_inf
import gradio as gr

shark_web = gr.Blocks()

with shark_web:
    gr.Markdown("Shark Models Demo.")
    with gr.Tabs():
        with gr.TabItem("ResNet50"):
            with gr.Group():
                image = gr.Image(label="Image")
                label = gr.Label(label="Output")
                resnet = gr.Button("Recognize Image")
                resnet.click(resnet_inf, inputs=image, outputs=label)
        with gr.TabItem("Albert MaskFill"):
            with gr.Group():
                masked_text = gr.Textbox(
                    label="Masked Text",
                    placeholder="Give me a sentence with [MASK] to fill",
                )
                decoded_res = gr.Label(label="Decoded Results")
                albert_mask = gr.Button("Decode Mask")
                albert_mask.click(
                    albert_maskfill_inf,
                    inputs=masked_text,
                    outputs=decoded_res,
                )
        with gr.TabItem("V-Diffusion"):
            with gr.Group():
                prompt = gr.Textbox(
                    label="Prompt", value="New York City, oil on canvas:5"
                )
                sample_count = gr.Number(label="Sample Count", value=1)
                batch_size = gr.Number(label="Batch Size", value=1)
                iters = gr.Number(label="Steps", value=2)
                device = gr.Textbox(label="Device", value="gpu")
                v_diffusion = gr.Button("Generate image from prompt")
                generated_img = gr.Image(type="pil", shape=(100, 100))
                v_diffusion.click(
                    vdiff_inf,
                    inputs=[prompt, sample_count, batch_size, iters, device],
                    outputs=generated_img,
                )

shark_web.launch(share=True, server_port=8080, enable_queue=True)
