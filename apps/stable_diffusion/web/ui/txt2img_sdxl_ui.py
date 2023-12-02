import os
import torch
import time
import sys
import gradio as gr
from PIL import Image
from math import ceil
from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    nodlogo_loc,
    get_custom_model_path,
    get_custom_model_files,
    scheduler_list,
    predefined_models,
    cancel_sd,
)
from apps.stable_diffusion.web.utils.metadata import import_png_metadata
from apps.stable_diffusion.web.utils.common_label_calc import status_label
from apps.stable_diffusion.src import (
    args,
    Text2ImageSDXLPipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
    save_output_img,
    prompt_examples,
    Image2ImagePipeline,
)
from apps.stable_diffusion.src.utils import (
    get_generated_imgs_path,
    get_generation_text_info,
)

# set initial values of iree_vulkan_target_triple, use_tuned and import_mlir.
init_iree_vulkan_target_triple = args.iree_vulkan_target_triple
init_iree_metal_target_platform = args.iree_metal_target_platform
init_use_tuned = args.use_tuned
init_import_mlir = args.import_mlir


def txt2img_sdxl_inf(
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    seed: str | int,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    model_id: str,
    precision: str,
    device: str,
    max_length: int,
    save_metadata_to_json: bool,
    save_metadata_to_png: bool,
    ondemand: bool,
    repeatable_seeds: bool,

):
    if precision != "fp16":
        print("currently we support fp16 for SDXL")
        precision = "fp16"
    from apps.stable_diffusion.web.ui.utils import (
        get_custom_model_pathfile,
        get_custom_vae_or_lora_weights,
        Config,
    )
    import apps.stable_diffusion.web.utils.global_obj as global_obj
    from apps.stable_diffusion.src.pipelines.pipeline_shark_stable_diffusion_utils import (
        SD_STATE_CANCEL,
    )

    args.prompts = [prompt]
    args.negative_prompts = [negative_prompt]
    args.guidance_scale = guidance_scale
    args.steps = steps
    args.scheduler = scheduler
    args.ondemand = ondemand

    # set ckpt_loc and hf_model_id.
    args.ckpt_loc = ""
    args.hf_model_id = ""
    args.custom_vae = ""

    # .safetensor or .chkpt on the custom model path
    if model_id in get_custom_model_files():
        args.ckpt_loc = get_custom_model_pathfile(model_id)
    # civitai download
    elif "civitai" in model_id:
        args.ckpt_loc = model_id
    # either predefined or huggingface
    else:
        args.hf_model_id = model_id

    # if custom_vae != "None":
    #     args.custom_vae = get_custom_model_pathfile(custom_vae, model="vae")

    args.save_metadata_to_json = save_metadata_to_json
    args.write_metadata_to_png = save_metadata_to_png

    args.use_lora = ""

    dtype = torch.float32 if precision == "fp32" else torch.half
    cpu_scheduling = not scheduler.startswith("Shark")
    new_config_obj = Config(
        "txt2img_sdxl",
        args.hf_model_id,
        args.ckpt_loc,
        args.custom_vae,
        precision,
        batch_size,
        max_length,
        height,
        width,
        device,
        use_lora=args.use_lora,
        stencils=None,
        ondemand=ondemand,
    )
    if (
        not global_obj.get_sd_obj()
        or global_obj.get_cfg_obj() != new_config_obj
    ):
        global_obj.clear_cache()
        global_obj.set_cfg_obj(new_config_obj)
        args.precision = precision
        args.batch_count = batch_count
        args.batch_size = batch_size
        args.max_length = max_length
        args.height = height
        args.width = width
        args.device = device.split("=>", 1)[1].strip()
        args.iree_vulkan_target_triple = init_iree_vulkan_target_triple
        args.iree_metal_target_platform = init_iree_metal_target_platform
        args.use_tuned = init_use_tuned
        args.import_mlir = init_import_mlir
        args.img_path = None
        set_init_device_flags()
        model_id = (
            args.hf_model_id
            if args.hf_model_id
            else "stabilityai/stable-diffusion-xl-base-1.0"
        )
        global_obj.set_schedulers(get_schedulers(model_id))
        scheduler_obj = global_obj.get_scheduler(scheduler)
        # For SDXL we set max_length as 77.
        print("Setting max_length = 77")
        max_length = 77
        if global_obj.get_cfg_obj().ondemand:
            print("Running txt2img in memory efficient mode.")
        txt2img_sdxl_obj = Text2ImageSDXLPipeline.from_pretrained(
            scheduler=scheduler_obj,
            import_mlir=args.import_mlir,
            model_id=args.hf_model_id,
            ckpt_loc=args.ckpt_loc,
            precision=precision,
            max_length=max_length,
            batch_size=batch_size,
            height=height,
            width=width,
            use_base_vae=args.use_base_vae,
            use_tuned=args.use_tuned,
            custom_vae=args.custom_vae,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            debug=args.import_debug if args.import_mlir else False,
            use_lora=args.use_lora,
            use_quantize=args.use_quantize,
            ondemand=global_obj.get_cfg_obj().ondemand,
        )
        global_obj.set_sd_obj(txt2img_sdxl_obj)

    global_obj.set_sd_scheduler(scheduler)

    start_time = time.time()
    global_obj.get_sd_obj().log = ""
    generated_imgs = []
    text_output = ""
    try:
        seeds = utils.batch_seeds(seed, batch_count, repeatable_seeds)
    except TypeError as error:
        raise gr.Error(str(error)) from None

    for current_batch in range(batch_count):
        out_imgs = global_obj.get_sd_obj().generate_images(
            prompt,
            negative_prompt,
            batch_size,
            height,
            width,
            steps,
            guidance_scale,
            seeds[current_batch],
            args.max_length,
            dtype,
            args.use_base_vae,
            cpu_scheduling,
            args.max_embeddings_multiples,
        )

        total_time = time.time() - start_time
        text_output = get_generation_text_info(
            seeds[: current_batch + 1], device
        )
        text_output += "\n" + global_obj.get_sd_obj().log
        text_output += f"\nTotal image(s) generation time: {total_time:.4f}sec"

        if global_obj.get_sd_status() == SD_STATE_CANCEL:
            break
        else:
            save_output_img(out_imgs[0], seeds[current_batch])
            generated_imgs.extend(out_imgs)
            yield generated_imgs, text_output, status_label(
                "Text-to-Image-SDXL",
                current_batch + 1,
                batch_count,
                batch_size,
            )

    return generated_imgs, text_output, ""


with gr.Blocks(title="Text-to-Image-SDXL") as txt2img_sdxl_web:
    with gr.Row(elem_id="ui_title"):
        nod_logo = Image.open(nodlogo_loc)
        with gr.Row():
            with gr.Column(scale=1, elem_id="demo_title_outer"):
                gr.Image(
                    value=nod_logo,
                    show_label=False,
                    interactive=False,
                    elem_id="top_logo",
                    width=150,
                    height=50,
                )
    with gr.Row(elem_id="ui_body"):
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                with gr.Row():
                    with gr.Column(scale=10):
                        with gr.Row():
                            t2i_model_info = f"Custom Model Path: {str(get_custom_model_path())}"
                            txt2img_sdxl_custom_model = gr.Dropdown(
                                label=f"Models",
                                info="Select, or enter HuggingFace Model ID or Civitai model download URL",
                                elem_id="custom_model",
                                value=os.path.basename(args.ckpt_loc)
                                if args.ckpt_loc
                                else "stabilityai/stable-diffusion-xl-base-1.0",
                                choices=[
                                    "stabilityai/stable-diffusion-xl-base-1.0"
                                ],
                                allow_custom_value=True,
                                scale=2,
                            )

                with gr.Group(elem_id="prompt_box_outer"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=args.prompts[0],
                        lines=2,
                        elem_id="prompt_box",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=args.negative_prompts[0],
                        lines=2,
                        elem_id="negative_prompt_box",
                    )

                with gr.Accordion(label="Advanced Options", open=False):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            elem_id="scheduler",
                            label="Scheduler",
                            value="DDIM",
                            choices=["DDIM"],
                            allow_custom_value=True,
                            visible=False,
                        )
                        with gr.Column():
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
                            1024,
                            value=1024,
                            step=8,
                            label="Height",
                            visible=False,
                        )
                        width = gr.Slider(
                            1024,
                            value=1024,
                            step=8,
                            label="Width",
                            visible=False,
                        )
                        precision = gr.Radio(
                            label="Precision",
                            value="fp16",
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
                        with gr.Column(scale=3):
                            steps = gr.Slider(
                                1, 100, value=args.steps, step=1, label="Steps"
                            )
                        with gr.Column(scale=3):
                            guidance_scale = gr.Slider(
                                0,
                                50,
                                value=args.guidance_scale,
                                step=0.1,
                                label="CFG Scale",
                            )
                        ondemand = gr.Checkbox(
                            value=args.ondemand,
                            label="Low VRAM",
                            interactive=True,
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
                        repeatable_seeds = gr.Checkbox(
                            args.repeatable_seeds,
                            label="Repeatable Seeds",
                        )
                with gr.Row():
                    seed = gr.Textbox(
                        value=args.seed,
                        label="Seed",
                        info="An integer or a JSON list of integers, -1 for random",
                    )
                    device = gr.Dropdown(
                        elem_id="device",
                        label="Device",
                        value=available_devices[0],
                        choices=available_devices,
                        allow_custom_value=True,
                    )
                with gr.Accordion(label="Prompt Examples!", open=False):
                    ex = gr.Examples(
                        examples=prompt_examples,
                        inputs=prompt,
                        cache_examples=False,
                        elem_id="prompt_examples",
                    )

            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    txt2img_sdxl_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                        columns=[2],
                        object_fit="contain",
                    )
                    std_output = gr.Textbox(
                        value=f"{t2i_model_info}\n"
                        f"Images will be saved at "
                        f"{get_generated_imgs_path()}",
                        lines=1,
                        elem_id="std_output",
                        show_label=False,
                    )
                    txt2img_sdxl_status = gr.Textbox(visible=False)
                with gr.Row():
                    stable_diffusion = gr.Button("Generate Image(s)")
                    random_seed = gr.Button("Randomize Seed")
                    random_seed.click(
                        lambda: -1,
                        inputs=[],
                        outputs=[seed],
                        queue=False,
                    )
                    stop_batch = gr.Button("Stop Batch")
                with gr.Row():
                    blank_thing_for_row = None

        kwargs = dict(
            fn=txt2img_sdxl_inf,
            inputs=[
                prompt,
                negative_prompt,
                height,
                width,
                steps,
                guidance_scale,
                seed,
                batch_count,
                batch_size,
                scheduler,
                txt2img_sdxl_custom_model,
                precision,
                device,
                max_length,
                save_metadata_to_json,
                save_metadata_to_png,
                ondemand,
                repeatable_seeds,
            ],
            outputs=[txt2img_sdxl_gallery, std_output, txt2img_sdxl_status],
            show_progress="minimal" if args.progress_bar else "none",
        )

        status_kwargs = dict(
            fn=lambda bc, bs: status_label("Text-to-Image-SDXL", 0, bc, bs),
            inputs=[batch_count, batch_size],
            outputs=txt2img_sdxl_status,
        )

        prompt_submit = prompt.submit(**status_kwargs).then(**kwargs)
        neg_prompt_submit = negative_prompt.submit(**status_kwargs).then(
            **kwargs
        )
        generate_click = stable_diffusion.click(**status_kwargs).then(**kwargs)
        stop_batch.click(
            fn=cancel_sd,
            cancels=[prompt_submit, neg_prompt_submit, generate_click],
        )
