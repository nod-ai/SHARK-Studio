from models.resnet50 import resnet_inf
from models.albert_maskfill import albert_maskfill_inf
from models.stable_diffusion import stable_diff_inf

#  from models.diffusion.v_diffusion import vdiff_inf
import gradio as gr

with gr.Blocks() as shark_web:
    gr.Markdown("Shark Models Demo.")
    with gr.Tabs():

        with gr.TabItem("ResNet50"):
            image = device = resnet = output = None
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    image = gr.Image(label="Image")
                    device = gr.Textbox(label="Device", value="cpu")
                    resnet = gr.Button("Recognize Image").style(
                        full_width=True
                    )
                with gr.Column(scale=1, min_width=600):
                    output = gr.Label(label="Output")
                    std_output = gr.Textbox(
                        label="Std Output", value="Nothing."
                    )
            resnet.click(
                resnet_inf,
                inputs=[image, device],
                outputs=[output, std_output],
            )

        with gr.TabItem("Albert MaskFill"):
            masked_text = device = albert_mask = decoded_res = None
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    masked_text = gr.Textbox(
                        label="Masked Text",
                        placeholder="Give me a sentence with [MASK] to fill",
                    )
                    device = gr.Textbox(label="Device", value="cpu")
                    albert_mask = gr.Button("Decode Mask")
                with gr.Column(scale=1, min_width=600):
                    decoded_res = gr.Label(label="Decoded Results")
                    std_output = gr.Textbox(
                        label="Std Output", value="Nothing."
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
            ) = mlir_loc = device = stable_diffusion = generated_img = None
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="a photograph of an astronaut riding a horse",
                    )
                    iters = gr.Number(label="Steps", value=2)
                    mlir_loc = gr.Textbox(
                        label="Location of MLIR(Relative to SHARK/web/)",
                        value="./models_mlir/stable_diffusion.mlir",
                    )
                    device = gr.Textbox(label="Device", value="vulkan")
                    stable_diffusion = gr.Button("Generate image from prompt")
                with gr.Column(scale=1, min_width=600):
                    generated_img = gr.Image(type="pil", shape=(100, 100))
                    std_output = gr.Textbox(
                        label="Std Output", value="Nothing."
                    )
            stable_diffusion.click(
                stable_diff_inf,
                inputs=[prompt, iters, mlir_loc, device],
                outputs=[generated_img, std_output],
            )

shark_web.launch(share=True, server_port=8080, enable_queue=True)
