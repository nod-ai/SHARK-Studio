import os
import torch
import time
import gradio as gr
from PIL import Image

from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    nodlogo_loc,
    get_custom_model_path,
    get_custom_model_files,
    scheduler_list_cpu_only,
    predefined_upscaler_models,
    cancel_sd,
)
from apps.stable_diffusion.web.ui.common_ui_events import (
    lora_changed,
    lora_strength_changed,
)
from apps.stable_diffusion.web.utils.common_label_calc import status_label
from apps.stable_diffusion.src import (
    args,
    UpscalerPipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
    save_output_img,
)
from apps.stable_diffusion.src.utils import get_generated_imgs_path

# set initial values of iree_vulkan_target_triple, use_tuned and import_mlir.
init_iree_vulkan_target_triple = args.iree_vulkan_target_triple
init_use_tuned = args.use_tuned
init_import_mlir = args.import_mlir


# Exposed to UI.
def upscaler_inf(
    prompt: str,
    negative_prompt: str,
    init_image,
    height: int,
    width: int,
    steps: int,
    noise_level: int,
    guidance_scale: float,
    seed: str,
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
    repeatable_seeds: bool,
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
    args.seed = seed
    args.steps = steps
    args.scheduler = scheduler
    args.ondemand = ondemand

    if init_image is None:
        return None, "An Initial Image is required"
    image = init_image.convert("RGB").resize((height, width))

    # set ckpt_loc and hf_model_id.
    args.ckpt_loc = ""
    args.hf_model_id = ""
    args.custom_vae = ""

    # .safetensor or .chkpt on the custom model path
    if model_id in get_custom_model_files(custom_checkpoint_type="upscaler"):
        args.ckpt_loc = get_custom_model_pathfile(model_id)
    # civitai download
    elif "civitai" in model_id:
        args.ckpt_loc = model_id
    # either predefined or huggingface
    else:
        args.hf_model_id = model_id

    if custom_vae != "None":
        args.custom_vae = get_custom_model_pathfile(custom_vae, model="vae")

    args.save_metadata_to_json = save_metadata_to_json
    args.write_metadata_to_png = save_metadata_to_png

    args.use_lora = get_custom_vae_or_lora_weights(lora_weights, "lora")
    args.lora_strength = lora_strength

    dtype = torch.float32 if precision == "fp32" else torch.half
    cpu_scheduling = not scheduler.startswith("Shark")
    args.height = 128
    args.width = 128
    new_config_obj = Config(
        "upscaler",
        args.hf_model_id,
        args.ckpt_loc,
        args.custom_vae,
        precision,
        batch_size,
        max_length,
        args.height,
        args.width,
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
        args.batch_size = batch_size
        args.max_length = max_length
        args.device = device.split("=>", 1)[1].strip()
        args.iree_vulkan_target_triple = init_iree_vulkan_target_triple
        args.use_tuned = init_use_tuned
        args.import_mlir = init_import_mlir
        set_init_device_flags()
        model_id = (
            args.hf_model_id
            if args.hf_model_id
            else "stabilityai/stable-diffusion-2-1-base"
        )
        global_obj.set_schedulers(get_schedulers(model_id))
        scheduler_obj = global_obj.get_scheduler(scheduler)
        global_obj.set_sd_obj(
            UpscalerPipeline.from_pretrained(
                scheduler_obj,
                args.import_mlir,
                args.hf_model_id,
                args.ckpt_loc,
                args.custom_vae,
                args.precision,
                args.max_length,
                args.batch_size,
                args.height,
                args.width,
                args.use_base_vae,
                args.use_tuned,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_lora=args.use_lora,
                lora_strength=args.lora_strength,
                ondemand=args.ondemand,
            )
        )

    global_obj.set_sd_scheduler(scheduler)
    global_obj.get_sd_obj().low_res_scheduler = global_obj.get_scheduler(
        "DDPM"
    )

    start_time = time.time()
    global_obj.get_sd_obj().log = ""
    generated_imgs = []
    extra_info = {"NOISE LEVEL": noise_level}
    try:
        seeds = utils.batch_seeds(seed, batch_count, repeatable_seeds)
    except TypeError as error:
        raise gr.Error(str(error)) from None

    for current_batch in range(batch_count):
        low_res_img = image
        high_res_img = Image.new("RGB", (height * 4, width * 4))

        for i in range(0, width, 128):
            for j in range(0, height, 128):
                box = (j, i, j + 128, i + 128)
                upscaled_image = global_obj.get_sd_obj().generate_images(
                    prompt,
                    negative_prompt,
                    low_res_img.crop(box),
                    batch_size,
                    args.height,
                    args.width,
                    steps,
                    noise_level,
                    guidance_scale,
                    seeds[current_batch],
                    args.max_length,
                    dtype,
                    args.use_base_vae,
                    cpu_scheduling,
                    args.max_embeddings_multiples,
                )
                if global_obj.get_sd_status() == SD_STATE_CANCEL:
                    break
                else:
                    high_res_img.paste(upscaled_image[0], (j * 4, i * 4))

            if global_obj.get_sd_status() == SD_STATE_CANCEL:
                break

        total_time = time.time() - start_time
        text_output = f"prompt={args.prompts}"
        text_output += f"\nnegative prompt={args.negative_prompts}"
        text_output += (
            f"\nmodel_id={args.hf_model_id}, " f"ckpt_loc={args.ckpt_loc}"
        )
        text_output += f"\nscheduler={args.scheduler}, " f"device={device}"
        text_output += (
            f"\nsteps={steps}, "
            f"noise_level={noise_level}, "
            f"guidance_scale={guidance_scale}, "
            f"seed={seeds[:current_batch + 1]}"
        )
        text_output += (
            f"\ninput size={height}x{width}, "
            f"output size={height*4}x{width*4}, "
            f"batch_count={batch_count}, "
            f"batch_size={batch_size}, "
            f"max_length={args.max_length}\n"
        )

        text_output += global_obj.get_sd_obj().log
        text_output += f"\nTotal image generation time: {total_time:.4f}sec"

        if global_obj.get_sd_status() == SD_STATE_CANCEL:
            break
        else:
            save_output_img(high_res_img, seeds[current_batch], extra_info)
            generated_imgs.append(high_res_img)
            global_obj.get_sd_obj().log += "\n"
            yield generated_imgs, text_output, status_label(
                "Upscaler", current_batch + 1, batch_count, batch_size
            )

    yield generated_imgs, text_output, ""


with gr.Blocks(title="Upscaler") as upscaler_web:
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
                upscaler_init_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    sources=["upload"],
                )
                with gr.Row():
                    upscaler_model_info = (
                        f"Custom Model Path: {str(get_custom_model_path())}"
                    )
                    upscaler_custom_model = gr.Dropdown(
                        label=f"Models",
                        info="Select, or enter HuggingFace Model ID or Civitai model download URL",
                        elem_id="custom_model",
                        value=(
                            os.path.basename(args.ckpt_loc)
                            if args.ckpt_loc
                            else "stabilityai/stable-diffusion-x4-upscaler"
                        ),
                        choices=get_custom_model_files(
                            custom_checkpoint_type="upscaler"
                        )
                        + predefined_upscaler_models,
                        allow_custom_value=True,
                        scale=2,
                    )
                    # janky fix for overflowing text
                    upscaler_vae_info = (
                        str(get_custom_model_path("vae"))
                    ).replace("\\", "\n\\")
                    upscaler_vae_info = f"VAE Path: {upscaler_vae_info}"
                    custom_vae = gr.Dropdown(
                        label=f"Custom VAE Models",
                        info=upscaler_vae_info,
                        elem_id="custom_model",
                        value=(
                            os.path.basename(args.custom_vae)
                            if args.custom_vae
                            else "None"
                        ),
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
                            value="DDIM",
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
                            128,
                            512,
                            value=args.height,
                            step=128,
                            label="Height",
                        )
                        width = gr.Slider(
                            128,
                            512,
                            value=args.width,
                            step=128,
                            label="Width",
                        )
                        precision = gr.Radio(
                            label="Precision",
                            value=args.precision,
                            choices=[
                                "fp16",
                                "fp32",
                            ],
                            visible=True,
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
                            1, 100, value=args.steps, step=1, label="Steps"
                        )
                        noise_level = gr.Slider(
                            0,
                            100,
                            value=args.noise_level,
                            step=1,
                            label="Noise Level",
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
                    upscaler_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                        columns=[2],
                        object_fit="contain",
                    )
                    std_output = gr.Textbox(
                        value=f"{upscaler_model_info}\n"
                        f"Images will be saved at "
                        f"{get_generated_imgs_path()}",
                        lines=2,
                        elem_id="std_output",
                        show_label=False,
                    )
                    upscaler_status = gr.Textbox(visible=False)
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
                    upscaler_sendto_img2img = gr.Button(value="SendTo Img2Img")
                    upscaler_sendto_inpaint = gr.Button(value="SendTo Inpaint")
                    upscaler_sendto_outpaint = gr.Button(
                        value="SendTo Outpaint"
                    )

        kwargs = dict(
            fn=upscaler_inf,
            inputs=[
                prompt,
                negative_prompt,
                upscaler_init_image,
                height,
                width,
                steps,
                noise_level,
                guidance_scale,
                seed,
                batch_count,
                batch_size,
                scheduler,
                upscaler_custom_model,
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
            outputs=[upscaler_gallery, std_output, upscaler_status],
            show_progress="minimal" if args.progress_bar else "none",
        )
        status_kwargs = dict(
            fn=lambda bc, bs: status_label("Upscaler", 0, bc, bs),
            inputs=[batch_count, batch_size],
            outputs=upscaler_status,
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
