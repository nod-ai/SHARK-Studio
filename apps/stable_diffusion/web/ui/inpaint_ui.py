from pathlib import Path
import gradio as gr
from PIL import Image
from apps.stable_diffusion.scripts import inpaint_inf
from apps.stable_diffusion.src import args
from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    nodlogo_loc,
    get_custom_model_path,
    get_custom_model_files,
    scheduler_list,
    predefined_paint_models,
)


with gr.Blocks(title="Inpainting") as inpaint_web:
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
                    custom_model = gr.Dropdown(
                        label=f"Models (Custom Model path: {get_custom_model_path()})",
                        value=args.ckpt_loc if args.ckpt_loc else "None",
                        choices=get_custom_model_files()
                        + predefined_paint_models,
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

                inpaint_init_image = gr.Image(
                    label="Masked Image",
                    source="upload",
                    tool="sketch",
                    type="pil",
                ).style(height=350)

                with gr.Accordion(label="Advanced Options", open=False):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            label="Scheduler",
                            value="PNDM",
                            choices=scheduler_list,
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
                        inpaint_full_res = gr.Radio(
                            choices=["Whole picture", "Only masked"],
                            type="index",
                            value="Whole picture",
                            label="Inpaint area",
                        )
                        inpaint_full_res_padding = gr.Slider(
                            minimum=0,
                            maximum=256,
                            step=4,
                            value=32,
                            label="Only masked padding, pixels",
                        )
                    with gr.Row():
                        steps = gr.Slider(
                            1, 100, value=args.steps, step=1, label="Steps"
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
                    inpaint_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=[2])
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
                    inpaint_sendto_img2img = gr.Button(value="SendTo Img2Img")
                    inpaint_sendto_outpaint = gr.Button(
                        value="SendTo Outpaint"
                    )

        kwargs = dict(
            fn=inpaint_inf,
            inputs=[
                prompt,
                negative_prompt,
                inpaint_init_image,
                height,
                width,
                inpaint_full_res,
                inpaint_full_res_padding,
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
            outputs=[inpaint_gallery, std_output],
            show_progress=args.progress_bar,
        )

        prompt.submit(**kwargs)
        negative_prompt.submit(**kwargs)
        stable_diffusion.click(**kwargs)
