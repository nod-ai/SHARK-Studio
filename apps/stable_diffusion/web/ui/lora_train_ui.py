from pathlib import Path
import os
import gradio as gr
from PIL import Image
from apps.stable_diffusion.scripts import lora_train
from apps.stable_diffusion.src import prompt_examples, args
from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    nodlogo_loc,
    get_custom_model_path,
    get_custom_model_files,
    get_custom_vae_or_lora_weights,
    scheduler_list,
    predefined_models,
)

with gr.Blocks(title="Lora Training") as lora_train_web:
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
                    with gr.Column(scale=10):
                        with gr.Row():
                            # janky fix for overflowing text
                            train_lora_model_info = (str(get_custom_model_path())).replace("\\",
                                                                                           "\n\\")
                            train_lora_model_info = f"Custom Model Path: {train_lora_model_info}"
                            custom_model = gr.Dropdown(
                                label=f"Models",
                                info=train_lora_model_info,
                                elem_id="custom_model",
                                value=os.path.basename(args.ckpt_loc)
                                if args.ckpt_loc
                                else "None",
                                choices=["None"]
                                + get_custom_model_files()
                                + predefined_models,
                            )
                            hf_model_id = gr.Textbox(
                                elem_id="hf_model_id",
                                placeholder="Select 'None' in the Models dropdown "
                                            "on the left and enter model ID here "
                                            "e.g: SG161222/Realistic_Vision_V1.3",
                                value="",
                                label="HuggingFace Model ID",
                                lines=3,
                            )

                with gr.Row():
                    # janky fix for overflowing text
                    train_lora_info = (str(get_custom_model_path('lora'))).replace("\\", "\n\\")
                    train_lora_info = f"LoRA Path: {train_lora_info}"
                    lora_weights = gr.Dropdown(
                        label=f"Standalone LoRA weights to initialize weights",
                        info=train_lora_info,
                        elem_id="lora_weights",
                        value="None",
                        choices=["None"] + get_custom_model_files("lora"),
                    )
                    lora_hf_id = gr.Textbox(
                        elem_id="lora_hf_id",
                        placeholder="Select 'None' in the Standalone LoRA weights "
                                    "dropdown on the left if you want to use a "
                                    "standalone HuggingFace model ID for LoRA here "
                                    "e.g: sayakpaul/sd-model-finetuned-lora-t4",
                        value="",
                        label="HuggingFace Model ID to initialize weights",
                        lines=3,
                    )
                with gr.Group(elem_id="image_dir_box_outer"):
                    training_images_dir = gr.Textbox(
                        label="ImageDirectory",
                        value=args.training_images_dir,
                        lines=1,
                        elem_id="prompt_box",
                    )
                with gr.Group(elem_id="prompt_box_outer"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=args.prompts[0],
                        lines=1,
                        elem_id="prompt_box",
                    )
                with gr.Accordion(label="Advanced Options", open=False):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            elem_id="scheduler",
                            label="Scheduler",
                            value=args.scheduler,
                            choices=scheduler_list,
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
                            1,
                            2000,
                            value=args.training_steps,
                            step=1,
                            label="Training Steps",
                        )
                        guidance_scale = gr.Slider(
                            0,
                            50,
                            value=args.guidance_scale,
                            step=0.1,
                            label="CFG Scale",
                        )
                    with gr.Row():
                        with gr.Column(scale=3):
                            batch_count = gr.Slider(
                                1,
                                100,
                                value=args.batch_count,
                                step=1,
                                label="Batch Count",
                                interactive=True,
                            )
                        with gr.Column(scale=3):
                            batch_size = gr.Slider(
                                1,
                                4,
                                value=args.batch_size,
                                step=1,
                                label="Batch Size",
                                interactive=True,
                            )
                        stop_batch = gr.Button("Stop Batch")
                with gr.Row():
                    seed = gr.Number(
                        value=args.seed, precision=0, label="Seed"
                    )
                    device = gr.Dropdown(
                        elem_id="device",
                        label="Device",
                        value=available_devices[0],
                        choices=available_devices,
                    )
                with gr.Row():
                    with gr.Column(scale=2):
                        random_seed = gr.Button("Randomize Seed")
                        random_seed.click(
                            lambda: -1,
                            inputs=[],
                            outputs=[seed],
                            queue=False,
                        )
                    with gr.Column(scale=6):
                        train_lora = gr.Button("Train LoRA")

                with gr.Accordion(label="Prompt Examples!", open=False):
                    ex = gr.Examples(
                        examples=prompt_examples,
                        inputs=prompt,
                        cache_examples=False,
                        elem_id="prompt_examples",
                    )

            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    std_output = gr.Textbox(
                        value="Nothing to show.",
                        lines=1,
                        show_label=False,
                    )
                lora_save_dir = (
                    args.lora_save_dir if args.lora_save_dir else Path.cwd()
                )
                lora_save_dir = Path(lora_save_dir, "lora")
                output_loc = gr.Textbox(
                    label="Saving Lora at",
                    value=lora_save_dir,
                )

        kwargs = dict(
            fn=lora_train,
            inputs=[
                prompt,
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
                training_images_dir,
                output_loc,
                get_custom_vae_or_lora_weights(
                    lora_weights, lora_hf_id, "lora"
                ),
            ],
            outputs=[std_output],
            show_progress="minimal" if args.progress_bar else "none",
        )

        prompt_submit = prompt.submit(**kwargs)
        train_click = train_lora.click(**kwargs)
        stop_batch.click(fn=None, cancels=[prompt_submit, train_click])
