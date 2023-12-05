import os
import torch
import time
import sys
import gradio as gr
import PIL.ImageOps
from PIL import Image

from gradio.components.image_editor import (
    Brush,
    Eraser,
    EditorData,
    EditorValue,
)
from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    nodlogo_loc,
    get_custom_model_path,
    get_custom_model_files,
    scheduler_list_cpu_only,
    predefined_paint_models,
    cancel_sd,
)
from apps.stable_diffusion.web.ui.common_ui_events import (
    lora_changed,
    lora_strength_changed,
)
from apps.stable_diffusion.src import (
    args,
    InpaintPipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
    clear_all,
    save_output_img,
)
from apps.stable_diffusion.src.utils import (
    get_generated_imgs_path,
    get_generation_text_info,
)
from apps.stable_diffusion.web.utils.common_label_calc import status_label


# set initial values of iree_vulkan_target_triple, use_tuned and import_mlir.
init_iree_vulkan_target_triple = args.iree_vulkan_target_triple
init_use_tuned = args.use_tuned
init_import_mlir = args.import_mlir


def set_image_states(editor_data):
    input_mask = editor_data["layers"][0]

    # inpaint_inf wants white mask on black background (?), whilst ImageEditor
    # delivers black mask on transparent (0 opacity) background
    inference_mask = Image.new(
        mode="RGB", size=input_mask.size, color=(255, 255, 255)
    )
    inference_mask.paste(input_mask, input_mask)
    inference_mask = PIL.ImageOps.invert(inference_mask)

    return (
        # we set the ImageEditor data again, because it likes to clear
        # the image layers (which include the mask) if the user hasn't
        # used the upload button, and we sent it and image
        # TODO: work out what is going wrong in that case so we don't have
        # to do this
        {
            "background": editor_data["background"],
            "layers": [input_mask],
            "composite": None,
        },
        editor_data["background"],
        input_mask,
        inference_mask,
    )


def reload_image_editor(editor_image, editor_mask):
    # we set the ImageEditor data again, because it likes to clear
    # the image layers (which include the mask) if the user hasn't
    # used the upload button, and we sent it the image
    # TODO: work out what is going wrong in that case so we don't have
    # to do this
    return {
        "background": editor_image,
        "layers": [editor_mask],
        "composite": None,
    }


# Exposed to UI.
def inpaint_inf(
    prompt: str,
    negative_prompt: str,
    image,
    mask_image,
    height: int,
    width: int,
    inpaint_full_res: bool,
    inpaint_full_res_padding: int,
    steps: int,
    guidance_scale: float,
    seed: str | int,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    model_id: str,
    custom_vae: str,
    precision: str,
    device: str,
    max_length: int,
    save_metadata_to_json: bool,
    save_metadata_to_png: bool,
    lora_weights: str,
    lora_strength: float,
    ondemand: bool,
    repeatable_seeds: int,
):
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
    args.img_path = "not none"
    args.mask_path = "not none"
    args.ondemand = ondemand

    # set ckpt_loc and hf_model_id.
    args.ckpt_loc = ""
    args.hf_model_id = ""
    args.custom_vae = ""

    # .safetensor or .chkpt on the custom model path
    if model_id in get_custom_model_files(custom_checkpoint_type="inpainting"):
        args.ckpt_loc = get_custom_model_pathfile(model_id)
    # civitai download
    elif "civitai" in model_id:
        args.ckpt_loc = model_id
    # either predefined or huggingface
    else:
        args.hf_model_id = model_id

    if custom_vae != "None":
        args.custom_vae = get_custom_model_pathfile(custom_vae, model="vae")

    args.use_lora = get_custom_vae_or_lora_weights(lora_weights, "lora")
    args.lora_strength = lora_strength

    args.save_metadata_to_json = save_metadata_to_json
    args.write_metadata_to_png = save_metadata_to_png

    dtype = torch.float32 if precision == "fp32" else torch.half
    cpu_scheduling = not scheduler.startswith("Shark")
    new_config_obj = Config(
        "inpaint",
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
        lora_strength=args.lora_strength,
        stencils=[],
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
        args.use_tuned = init_use_tuned
        args.import_mlir = init_import_mlir
        set_init_device_flags()
        model_id = (
            args.hf_model_id
            if args.hf_model_id
            else "stabilityai/stable-diffusion-2-inpainting"
        )
        global_obj.set_schedulers(get_schedulers(model_id))
        scheduler_obj = global_obj.get_scheduler(scheduler)
        global_obj.set_sd_obj(
            InpaintPipeline.from_pretrained(
                scheduler=scheduler_obj,
                import_mlir=args.import_mlir,
                model_id=args.hf_model_id,
                ckpt_loc=args.ckpt_loc,
                custom_vae=args.custom_vae,
                precision=args.precision,
                max_length=args.max_length,
                batch_size=args.batch_size,
                height=args.height,
                width=args.width,
                use_base_vae=args.use_base_vae,
                use_tuned=args.use_tuned,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                debug=args.import_debug if args.import_mlir else False,
                use_lora=args.use_lora,
                lora_strength=args.lora_strength,
                ondemand=args.ondemand,
            )
        )

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
            image,
            mask_image,
            batch_size,
            height,
            width,
            inpaint_full_res,
            inpaint_full_res_padding,
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
                "Inpaint", current_batch + 1, batch_count, batch_size
            )

    return generated_imgs, text_output


with gr.Blocks(title="Inpainting") as inpaint_web:
    editor_image = gr.State()
    editor_mask = gr.State()
    inference_mask = gr.State()
    with gr.Row(elem_id="ui_title"):
        nod_logo = Image.open(nodlogo_loc)
        with gr.Row():
            with gr.Column(scale=1, elem_id="demo_title_outer"):
                gr.Image(
                    value=nod_logo,
                    show_label=False,
                    interactive=False,
                    show_download_button=False,
                    elem_id="top_logo",
                    width=150,
                    height=50,
                )
    with gr.Row(elem_id="ui_body"):
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                inpaint_init_image = gr.Sketchpad(
                    label="Masked Image",
                    type="pil",
                    sources=("clipboard", "upload"),
                    interactive=True,
                    brush=Brush(
                        colors=["#000000"],
                        color_mode="fixed",
                    ),
                )
                with gr.Row():
                    # janky fix for overflowing text
                    inpaint_model_info = (
                        f"Custom Model Path: {str(get_custom_model_path())}"
                    )
                    inpaint_custom_model = gr.Dropdown(
                        label=f"Models",
                        info="Select, or enter HuggingFace Model ID or Civitai model download URL",
                        elem_id="custom_model",
                        value=os.path.basename(args.ckpt_loc)
                        if args.ckpt_loc
                        else "stabilityai/stable-diffusion-2-inpainting",
                        choices=get_custom_model_files(
                            custom_checkpoint_type="inpainting"
                        )
                        + predefined_paint_models,
                        allow_custom_value=True,
                        scale=2,
                    )
                    # janky fix for overflowing text
                    inpaint_vae_info = (
                        str(get_custom_model_path("vae"))
                    ).replace("\\", "\n\\")
                    inpaint_vae_info = f"VAE Path: {inpaint_vae_info}"
                    custom_vae = gr.Dropdown(
                        label=f"Custom VAE Models",
                        info=inpaint_vae_info,
                        elem_id="custom_model",
                        value=os.path.basename(args.custom_vae)
                        if args.custom_vae
                        else "None",
                        choices=["None"] + get_custom_model_files("vae"),
                        allow_custom_value=True,
                        scale=1,
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
                with gr.Accordion(label="LoRA Options", open=False):
                    with gr.Row():
                        lora_weights = gr.Dropdown(
                            label=f"LoRA Weights",
                            info=f"Select from LoRA in {str(get_custom_model_path('lora'))}, or enter HuggingFace Model ID",
                            elem_id="lora_weights",
                            value="None",
                            choices=["None"] + get_custom_model_files("lora"),
                            allow_custom_value=True,
                            scale=3,
                        )
                        lora_strength = gr.Number(
                            label="LoRA Strength",
                            info="Will be baked into the .vmfb",
                            step=0.01,
                            # number is checked on change so to allow 0.n values
                            # we have to allow 0 or you can't type 0.n in
                            minimum=0.0,
                            maximum=2.0,
                            value=args.lora_strength,
                            scale=1,
                        )
                    with gr.Row():
                        lora_tags = gr.HTML(
                            value="<div><i>No LoRA selected</i></div>",
                            elem_classes="lora-tags",
                        )
                with gr.Accordion(label="Advanced Options", open=False):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            elem_id="scheduler",
                            label="Scheduler",
                            value="EulerDiscrete",
                            choices=scheduler_list_cpu_only,
                            allow_custom_value=True,
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
                        ondemand = gr.Checkbox(
                            value=args.ondemand,
                            label="Low VRAM",
                            interactive=True,
                        )
                    with gr.Row():
                        with gr.Column(scale=3):
                            guidance_scale = gr.Slider(
                                0,
                                50,
                                value=args.guidance_scale,
                                step=0.1,
                                label="CFG Scale",
                            )
                        with gr.Column(scale=3):
                            batch_count = gr.Slider(
                                1,
                                100,
                                value=args.batch_count,
                                step=1,
                                label="Batch Count",
                                interactive=True,
                            )
                        repeatable_seeds = gr.Checkbox(
                            args.repeatable_seeds,
                            label="Repeatable Seeds",
                        )
                    with gr.Row():
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

            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    inpaint_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                        columns=[2],
                        object_fit="contain",
                        # TODO: Re-enable download when fixed in Gradio
                        show_download_button=False,
                    )
                    std_output = gr.Textbox(
                        value=f"{inpaint_model_info}\n"
                        "Images will be saved at "
                        f"{get_generated_imgs_path()}",
                        lines=2,
                        elem_id="std_output",
                        show_label=False,
                    )
                    inpaint_status = gr.Textbox(visible=False)
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
                with gr.Row():
                    inpaint_sendto_img2img = gr.Button(value="SendTo Img2Img")
                    inpaint_sendto_outpaint = gr.Button(
                        value="SendTo Outpaint"
                    )
                    inpaint_sendto_upscaler = gr.Button(
                        value="SendTo Upscaler"
                    )

        kwargs = dict(
            fn=inpaint_inf,
            inputs=[
                prompt,
                negative_prompt,
                editor_image,
                inference_mask,
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
                inpaint_custom_model,
                custom_vae,
                precision,
                device,
                max_length,
                save_metadata_to_json,
                save_metadata_to_png,
                lora_weights,
                lora_strength,
                ondemand,
                repeatable_seeds,
            ],
            outputs=[inpaint_gallery, std_output, inpaint_status],
            show_progress="minimal" if args.progress_bar else "none",
        )
        status_kwargs = dict(
            fn=lambda bc, bs: status_label("Inpaint", 0, bc, bs),
            inputs=[batch_count, batch_size],
            outputs=inpaint_status,
            show_progress="none",
        )
        set_image_states_args = dict(
            fn=set_image_states,
            inputs=[inpaint_init_image],
            outputs=[
                inpaint_init_image,
                editor_image,
                editor_mask,
                inference_mask,
            ],
            show_progress="none",
        )
        reload_image_editor_args = dict(
            fn=reload_image_editor,
            inputs=[editor_image, editor_mask],
            outputs=[inpaint_init_image],
            show_progress="none",
        )

        # all these trigger generation
        prompt_submit = (
            prompt.submit(**set_image_states_args)
            .then(**status_kwargs)
            .then(**kwargs)
            .then(**reload_image_editor_args)
        )
        neg_prompt_submit = (
            negative_prompt.submit(**set_image_states_args)
            .then(**status_kwargs)
            .then(**kwargs)
            .then(**reload_image_editor_args)
        )
        generate_click = (
            stable_diffusion.click(**set_image_states_args)
            .then(**status_kwargs)
            .then(**kwargs)
            .then(**reload_image_editor_args)
        )

        # Attempts to cancel generation
        stop_batch.click(
            fn=cancel_sd,
            cancels=[prompt_submit, neg_prompt_submit, generate_click],
        )

        # Updates LoRA information when one is selected
        lora_weights.change(
            fn=lora_changed,
            inputs=[lora_weights],
            outputs=[lora_tags],
            queue=True,
        )

        lora_strength.change(
            fn=lora_strength_changed,
            inputs=lora_strength,
            outputs=lora_strength,
            queue=False,
            show_progress=False,
        )
