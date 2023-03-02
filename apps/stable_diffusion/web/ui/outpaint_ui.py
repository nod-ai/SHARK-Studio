import os
import sys
import glob
from pathlib import Path
import gradio as gr
from PIL import Image
from apps.stable_diffusion.scripts import outpaint_inf
from apps.stable_diffusion.src import args
from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    nodlogo_loc,
)


with gr.Blocks(title="Outpainting") as outpaint_web:
    with gr.Row(elem_id="ui_title"):
        nod_logo = Image.open(nodlogo_loc)
        with gr.Row():
            with gr.Column(scale=1, elem_id="demo_title_outer"):
                gr.Image(
                    value=nod_logo,
                    show_label=False,
                    interactive=False,
                    elem_id="top_logo",
                ).style(width=150, height=50)
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
                        value=args.ckpt_loc if args.ckpt_loc else "None",
                        choices=ckpt_files
                        + [
                            "runwayml/stable-diffusion-inpainting",
                            "stabilityai/stable-diffusion-2-inpainting",
                        ],
                    )
                    hf_model_id = gr.Textbox(
                        placeholder="Select 'None' in the Models dropdown on the left and enter model ID here e.g: ghunkins/stable-diffusion-liberty-inpainting",
                        value="",
                        label="HuggingFace Model ID",
                        lines=3,
                    )

                with gr.Group(elem_id="prompt_box_outer"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=args.prompts[0],
                        lines=1,
                        elem_id="prompt_box",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=args.negative_prompts[0],
                        lines=1,
                        elem_id="negative_prompt_box",
                    )

                outpaint_init_image = gr.Image(
                    label="Input Image", type="pil"
                ).style(height=300)

                with gr.Accordion(label="Advanced Options", open=False):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            label="Scheduler",
                            value="PNDM",
                            choices=[
                                "DDIM",
                                "PNDM",
                                "DPMSolverMultistep",
                                "EulerAncestralDiscrete",
                            ],
                        )
                        with gr.Group():
                            save_metadata_to_png = gr.Checkbox(
                                label="Save prompt information to PNG",
                                value=args.write_metadata_to_png,
                                interactive=True,
                            )
                            save_metadata_to_json = gr.Checkbox(
                                label="Save prompt information to JSON file",
                                value=args.save_metadata_to_json,
                                interactive=True,
                            )
                    with gr.Row():
                        pixels = gr.Slider(
                            8,
                            256,
                            value=args.pixels,
                            step=8,
                            label="Pixels to expand",
                        )
                        mask_blur = gr.Slider(
                            0,
                            64,
                            value=args.mask_blur,
                            step=1,
                            label="Mask blur",
                        )
                    with gr.Row():
                        directions = gr.CheckboxGroup(
                            label="Outpainting direction",
                            choices=["left", "right", "up", "down"],
                            value=["left", "right", "up", "down"],
                        )
                    with gr.Row():
                        noise_q = gr.Slider(
                            0.0,
                            4.0,
                            value=1.0,
                            step=0.01,
                            label="Fall-off exponent (lower=higher detail)",
                        )
                        color_variation = gr.Slider(
                            0.0,
                            1.0,
                            value=0.05,
                            step=0.01,
                            label="Color variation",
                        )
                    with gr.Row():
                        height = gr.Slider(
                            384, 768, value=args.height, step=8, label="Height"
                        )
                        width = gr.Slider(
                            384, 768, value=args.width, step=8, label="Width"
                        )
                        precision = gr.Radio(
                            label="Precision",
                            value=args.precision,
                            choices=[
                                "fp16",
                                "fp32",
                            ],
                            visible=False,
                        )
                        max_length = gr.Radio(
                            label="Max Length",
                            value=args.max_length,
                            choices=[
                                64,
                                77,
                            ],
                            visible=False,
                        )
                    with gr.Row():
                        steps = gr.Slider(
                            1, 100, value=20, step=1, label="Steps"
                        )
                    with gr.Row():
                        guidance_scale = gr.Slider(
                            0,
                            50,
                            value=args.guidance_scale,
                            step=0.1,
                            label="CFG Scale",
                        )
                        batch_count = gr.Slider(
                            1,
                            100,
                            value=args.batch_count,
                            step=1,
                            label="Batch Count",
                            interactive=True,
                        )
                        batch_size = gr.Slider(
                            1,
                            4,
                            value=args.batch_size,
                            step=1,
                            label="Batch Size",
                            interactive=False,
                            visible=False,
                        )
                with gr.Row():
                    seed = gr.Number(
                        value=args.seed, precision=0, label="Seed"
                    )
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
                    stable_diffusion = gr.Button("Generate Image(s)")

            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    outpaint_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=[2])
                    outpaint_output = gr.Image(
                        visible=False,
                    )
                    std_output = gr.Textbox(
                        value="Nothing to show.",
                        lines=1,
                        show_label=False,
                    )
                output_dir = args.output_dir if args.output_dir else Path.cwd()
                output_dir = Path(output_dir, "generated_imgs")
                output_loc = gr.Textbox(
                    label="Saving Images at",
                    value=output_dir,
                    interactive=False,
                )
                with gr.Row():
                    outpaint_sendto_img2img = gr.Button(value="SendTo Img2Img")
                    outpaint_sendto_inpaint = gr.Button(value="SendTo Inpaint")

        kwargs = dict(
            fn=outpaint_inf,
            inputs=[
                prompt,
                negative_prompt,
                outpaint_init_image,
                pixels,
                mask_blur,
                directions,
                noise_q,
                color_variation,
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
            outputs=[outpaint_gallery, outpaint_output, std_output],
            show_progress=args.progress_bar,
        )

        prompt.submit(**kwargs)
        negative_prompt.submit(**kwargs)
        stable_diffusion.click(**kwargs)
