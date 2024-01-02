import json
import os
import warnings
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
    scheduler_list_cpu_only,
    predefined_models,
    cancel_sd,
)
from apps.stable_diffusion.web.ui.common_ui_events import lora_changed
from apps.stable_diffusion.web.utils.metadata import import_png_metadata
from apps.stable_diffusion.web.utils.common_label_calc import status_label
from apps.stable_diffusion.src import (
    args,
    Text2ImagePipeline,
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
    resampler_list,
)

# Names of all interactive fields that can be edited by user
all_gradio_labels = [
    "txt2img_custom_model",
    "custom_vae",
    "prompt",
    "negative_prompt",
    "lora_weights",
    "lora_hf_id",
    "scheduler",
    "save_metadata_to_png",
    "save_metadata_to_json",
    "height",
    "width",
    "steps",
    "guidance_scale",
    "Low VRAM",
    "use_hiresfix",
    "resample_type",
    "hiresfix_height",
    "hiresfix_width",
    "hiresfix_strength",
    "batch_count",
    "batch_size",
    "repeatable_seeds",
    "seed",
    "device",
]

# set initial values of iree_vulkan_target_triple, use_tuned and import_mlir.
init_iree_vulkan_target_triple = args.iree_vulkan_target_triple
init_iree_metal_target_platform = args.iree_metal_target_platform
init_use_tuned = args.use_tuned
init_import_mlir = args.import_mlir


def txt2img_inf(
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
    custom_vae: str,
    precision: str,
    device: str,
    max_length: int,
    save_metadata_to_json: bool,
    save_metadata_to_png: bool,
    lora_weights: str,
    lora_hf_id: str,
    ondemand: bool,
    repeatable_seeds: bool,
    use_hiresfix: bool,
    hiresfix_height: int,
    hiresfix_width: int,
    hiresfix_strength: float,
    resample_type: str,
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

    if custom_vae != "None":
        args.custom_vae = get_custom_model_pathfile(custom_vae, model="vae")

    args.save_metadata_to_json = save_metadata_to_json
    args.write_metadata_to_png = save_metadata_to_png

    args.use_lora = get_custom_vae_or_lora_weights(
        lora_weights, lora_hf_id, "lora"
    )

    dtype = torch.float32 if precision == "fp32" else torch.half
    cpu_scheduling = not scheduler.startswith("Shark")
    new_config_obj = Config(
        "txt2img",
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
        args.use_hiresfix = use_hiresfix
        args.hiresfix_height = hiresfix_height
        args.hiresfix_width = hiresfix_width
        args.hiresfix_strength = hiresfix_strength
        args.resample_type = resample_type
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
            else "stabilityai/stable-diffusion-2-1-base"
        )
        global_obj.set_schedulers(get_schedulers(model_id))
        scheduler_obj = global_obj.get_scheduler(scheduler)
        global_obj.set_sd_obj(
            Text2ImagePipeline.from_pretrained(
                scheduler=scheduler_obj,
                import_mlir=args.import_mlir,
                model_id=args.hf_model_id,
                ckpt_loc=args.ckpt_loc,
                precision=args.precision,
                max_length=args.max_length,
                batch_size=args.batch_size,
                height=args.height,
                width=args.width,
                use_base_vae=args.use_base_vae,
                use_tuned=args.use_tuned,
                custom_vae=args.custom_vae,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                debug=args.import_debug if args.import_mlir else False,
                use_lora=args.use_lora,
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
        # TODO: allow user to save original image
        # TODO: add option to let user keep both pipelines loaded, and unload
        #  either at will
        # TODO: add custom step value slider
        # TODO: add option to use secondary model for the img2img pass
        if use_hiresfix is True:
            new_config_obj = Config(
                "img2img",
                args.hf_model_id,
                args.ckpt_loc,
                args.custom_vae,
                precision,
                1,
                max_length,
                height,
                width,
                device,
                use_lora=args.use_lora,
                stencils=[],
                ondemand=ondemand,
            )

            global_obj.clear_cache()
            global_obj.set_cfg_obj(new_config_obj)
            set_init_device_flags()
            model_id = (
                args.hf_model_id
                if args.hf_model_id
                else "stabilityai/stable-diffusion-2-1-base"
            )
            global_obj.set_schedulers(get_schedulers(model_id))
            scheduler_obj = global_obj.get_scheduler(args.scheduler)

            global_obj.set_sd_obj(
                Image2ImagePipeline.from_pretrained(
                    scheduler_obj,
                    args.import_mlir,
                    args.hf_model_id,
                    args.ckpt_loc,
                    args.custom_vae,
                    args.precision,
                    args.max_length,
                    1,
                    hiresfix_height,
                    hiresfix_width,
                    args.use_base_vae,
                    args.use_tuned,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    debug=args.import_debug if args.import_mlir else False,
                    use_lora=args.use_lora,
                    ondemand=args.ondemand,
                )
            )

            global_obj.set_sd_scheduler(args.scheduler)

            out_imgs = global_obj.get_sd_obj().generate_images(
                prompt,
                negative_prompt,
                out_imgs[0],
                batch_size,
                hiresfix_height,
                hiresfix_width,
                ceil(steps / hiresfix_strength),
                hiresfix_strength,
                guidance_scale,
                seeds[current_batch],
                args.max_length,
                dtype,
                args.use_base_vae,
                cpu_scheduling,
                args.max_embeddings_multiples,
                stencils=[],
                control_mode=None,
                resample_type=resample_type,
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
                "Text-to-Image", current_batch + 1, batch_count, batch_size
            )

    return generated_imgs, text_output, ""


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_path, relative_path)


dark_theme = resource_path("ui/css/sd_dark_theme.css")


# This function export values for all fields that can be edited by user to the settings.json file in ui folder
def export_settings(*values):
    settings_list = list(zip(all_gradio_labels, values))
    settings = {}

    for label, value in settings_list:
        settings[label] = value

    settings = {"txt2img": settings}
    with open("./ui/settings.json", "w") as json_file:
        json.dump(settings, json_file, indent=4)


# This function loads all values for all fields that can be edited by user from the settings.json file in ui folder
def load_settings():
    try:
        with open("./ui/settings.json", "r") as json_file:
            loaded_settings = json.load(json_file)["txt2img"]
    except (FileNotFoundError, KeyError):
        warnings.warn(
            "Settings.json file not found or 'txt2img' key is missing. Using default values for fields."
        )
        loaded_settings = (
            {}
        )  # json file not existing or the data wasn't saved yet

    return [
        loaded_settings.get(
            "txt2img_custom_model",
            os.path.basename(args.ckpt_loc)
            if args.ckpt_loc
            else "stabilityai/stable-diffusion-2-1-base",
        ),
        loaded_settings.get(
            "custom_vae",
            os.path.basename(args.custom_vae) if args.custom_vae else "None",
        ),
        loaded_settings.get("prompt", args.prompts[0]),
        loaded_settings.get("negative_prompt", args.negative_prompts[0]),
        loaded_settings.get("lora_weights", "None"),
        loaded_settings.get("lora_hf_id", ""),
        loaded_settings.get("scheduler", args.scheduler),
        loaded_settings.get(
            "save_metadata_to_png", args.write_metadata_to_png
        ),
        loaded_settings.get(
            "save_metadata_to_json", args.save_metadata_to_json
        ),
        loaded_settings.get("height", args.height),
        loaded_settings.get("width", args.width),
        loaded_settings.get("steps", args.steps),
        loaded_settings.get("guidance_scale", args.guidance_scale),
        loaded_settings.get("Low VRAM", args.ondemand),
        loaded_settings.get("use_hiresfix", args.use_hiresfix),
        loaded_settings.get("resample_type", args.resample_type),
        loaded_settings.get("hiresfix_height", args.hiresfix_height),
        loaded_settings.get("hiresfix_width", args.hiresfix_width),
        loaded_settings.get("hiresfix_strength", args.hiresfix_strength),
        loaded_settings.get("batch_count", args.batch_count),
        loaded_settings.get("batch_size", args.batch_size),
        loaded_settings.get("repeatable_seeds", args.repeatable_seeds),
        loaded_settings.get("seed", args.seed),
        loaded_settings.get("device", available_devices[0]),
    ]


# This function loads the user's exported default settings on the start of program
def onload_load_settings():
    loaded_data = load_settings()
    structured_data = settings_list = list(zip(all_gradio_labels, loaded_data))
    return dict(structured_data)


default_settings = onload_load_settings()
with gr.Blocks(title="Text-to-Image", css=dark_theme) as txt2img_web:
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
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            t2i_model_info = f"Custom Model Path: {str(get_custom_model_path())}"
                            txt2img_custom_model = gr.Dropdown(
                                label=f"Models",
                                info="Select, or enter HuggingFace Model ID or Civitai model download URL",
                                elem_id="custom_model",
                                value=default_settings.get(
                                    "txt2img_custom_model"
                                ),
                                choices=get_custom_model_files()
                                + predefined_models,
                                allow_custom_value=True,
                                scale=11,
                            )
                            # janky fix for overflowing text
                            t2i_vae_info = (
                                str(get_custom_model_path("vae"))
                            ).replace("\\", "\n\\")
                            t2i_vae_info = f"VAE Path: {t2i_vae_info}"
                            custom_vae = gr.Dropdown(
                                label=f"VAE Models",
                                info=t2i_vae_info,
                                elem_id="custom_model",
                                value=default_settings.get("custom_vae"),
                                choices=["None"]
                                + get_custom_model_files("vae"),
                                allow_custom_value=True,
                                scale=4,
                            )
                            txt2img_png_info_img = gr.Image(
                                label="Import PNG info",
                                elem_id="txt2img_prompt_image",
                                type="pil",
                                visible=True,
                                sources=["upload"],
                                scale=1,
                            )
                with gr.Group(elem_id="prompt_box_outer"):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=default_settings.get("prompt"),
                        lines=2,
                        elem_id="prompt_box",
                    )
                    # TODO: coming soon
                    autogen = gr.Checkbox(
                        label="Continuous Generation",
                        visible=False,
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=default_settings.get("negative_prompt"),
                        lines=2,
                        elem_id="negative_prompt_box",
                    )
                with gr.Accordion(label="LoRA Options", open=False):
                    with gr.Row():
                        # janky fix for overflowing text
                        t2i_lora_info = (
                            str(get_custom_model_path("lora"))
                        ).replace("\\", "\n\\")
                        t2i_lora_info = f"LoRA Path: {t2i_lora_info}"
                        lora_weights = gr.Dropdown(
                            label=f"Standalone LoRA Weights",
                            info=t2i_lora_info,
                            elem_id="lora_weights",
                            value=default_settings.get("lora_weights"),
                            choices=["None"] + get_custom_model_files("lora"),
                            allow_custom_value=True,
                        )
                        lora_hf_id = gr.Textbox(
                            elem_id="lora_hf_id",
                            placeholder="Select 'None' in the Standalone LoRA "
                            "weights dropdown on the left if you want to use "
                            "a standalone HuggingFace model ID for LoRA here "
                            "e.g: sayakpaul/sd-model-finetuned-lora-t4",
                            value=default_settings.get("lora_hf_id"),
                            label="HuggingFace Model ID",
                            lines=3,
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
                            value=default_settings.get("scheduler"),
                            choices=scheduler_list,
                            allow_custom_value=True,
                        )
                        with gr.Column():
                            save_metadata_to_png = gr.Checkbox(
                                label="Save prompt information to PNG",
                                value=default_settings.get(
                                    "save_metadata_to_png"
                                ),
                                interactive=True,
                            )
                            save_metadata_to_json = gr.Checkbox(
                                label="Save prompt information to JSON file",
                                value=default_settings.get(
                                    "save_metadata_to_json"
                                ),
                                interactive=True,
                            )
                    with gr.Row():
                        height = gr.Slider(
                            384,
                            768,
                            value=default_settings.get("height"),
                            step=8,
                            label="Height",
                        )
                        width = gr.Slider(
                            384,
                            768,
                            value=default_settings.get("width"),
                            step=8,
                            label="Width",
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
                        with gr.Column(scale=3):
                            steps = gr.Slider(
                                1,
                                100,
                                value=default_settings.get("steps"),
                                step=1,
                                label="Steps",
                            )
                        with gr.Column(scale=3):
                            guidance_scale = gr.Slider(
                                0,
                                50,
                                value=default_settings.get("guidance_scale"),
                                step=0.1,
                                label="CFG Scale",
                            )
                        ondemand = gr.Checkbox(
                            value=default_settings.get("Low VRAM"),
                            label="Low VRAM",
                            interactive=True,
                        )
                    with gr.Row():
                        with gr.Column(scale=3):
                            batch_count = gr.Slider(
                                1,
                                100,
                                value=default_settings.get("batch_count"),
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
                                label=default_settings.get("batch_size"),
                                interactive=True,
                            )
                        repeatable_seeds = gr.Checkbox(
                            default_settings.get("repeatable_seeds"),
                            label="Repeatable Seeds",
                        )
                with gr.Accordion(label="Hires Fix Options", open=False):
                    with gr.Group():
                        with gr.Row():
                            use_hiresfix = gr.Checkbox(
                                value=default_settings.get("use_hiresfix"),
                                label="Use Hires Fix",
                                interactive=True,
                            )
                            resample_type = gr.Dropdown(
                                value=default_settings.get("resample_type"),
                                choices=resampler_list,
                                label="Resample Type",
                                allow_custom_value=False,
                            )
                        hiresfix_height = gr.Slider(
                            384,
                            768,
                            value=default_settings.get("hiresfix_height"),
                            step=8,
                            label="Hires Fix Height",
                        )
                        hiresfix_width = gr.Slider(
                            384,
                            768,
                            value=default_settings.get("hiresfix_width"),
                            step=8,
                            label="Hires Fix Width",
                        )
                        hiresfix_strength = gr.Slider(
                            0,
                            1,
                            value=default_settings.get("hiresfix_strength"),
                            step=0.01,
                            label="Hires Fix Denoising Strength",
                        )
                with gr.Row():
                    seed = gr.Textbox(
                        value=default_settings.get("seed"),
                        label="Seed",
                        info="An integer or a JSON list of integers, -1 for random",
                    )
                    device = gr.Dropdown(
                        elem_id="device",
                        label="Device",
                        value=default_settings.get("device"),
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
                    txt2img_gallery = gr.Gallery(
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
                    txt2img_status = gr.Textbox(visible=False)
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
                    txt2img_sendto_img2img = gr.Button(value="SendTo Img2Img")
                    txt2img_sendto_inpaint = gr.Button(value="SendTo Inpaint")
                    txt2img_sendto_outpaint = gr.Button(
                        value="SendTo Outpaint"
                    )
                    txt2img_sendto_upscaler = gr.Button(
                        value="SendTo Upscaler"
                    )
                with gr.Row():
                    with gr.Column(scale=2):
                        export_defaults = gr.Button(
                            value="Load Default Settings"
                        )
                        export_defaults.click(
                            fn=load_settings,
                            inputs=[],
                            outputs=[
                                txt2img_custom_model,
                                custom_vae,
                                prompt,
                                negative_prompt,
                                lora_weights,
                                lora_hf_id,
                                scheduler,
                                save_metadata_to_png,
                                save_metadata_to_json,
                                height,
                                width,
                                steps,
                                guidance_scale,
                                ondemand,
                                use_hiresfix,
                                resample_type,
                                hiresfix_height,
                                hiresfix_width,
                                hiresfix_strength,
                                batch_count,
                                batch_size,
                                repeatable_seeds,
                                seed,
                                device,
                            ],
                        )
                    with gr.Column(scale=2):
                        export_defaults = gr.Button(
                            value="Export Default Settings"
                        )
                        export_defaults.click(
                            fn=export_settings,
                            inputs=[
                                txt2img_custom_model,
                                custom_vae,
                                prompt,
                                negative_prompt,
                                lora_weights,
                                lora_hf_id,
                                scheduler,
                                save_metadata_to_png,
                                save_metadata_to_json,
                                height,
                                width,
                                steps,
                                guidance_scale,
                                ondemand,
                                use_hiresfix,
                                resample_type,
                                hiresfix_height,
                                hiresfix_width,
                                hiresfix_strength,
                                batch_count,
                                batch_size,
                                repeatable_seeds,
                                seed,
                                device,
                            ],
                            outputs=[],
                        )

        kwargs = dict(
            fn=txt2img_inf,
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
                txt2img_custom_model,
                custom_vae,
                precision,
                device,
                max_length,
                save_metadata_to_json,
                save_metadata_to_png,
                lora_weights,
                lora_hf_id,
                ondemand,
                repeatable_seeds,
                use_hiresfix,
                hiresfix_height,
                hiresfix_width,
                hiresfix_strength,
                resample_type,
            ],
            outputs=[txt2img_gallery, std_output, txt2img_status],
            show_progress="minimal" if args.progress_bar else "none",
        )

        status_kwargs = dict(
            fn=lambda bc, bs: status_label("Text-to-Image", 0, bc, bs),
            inputs=[batch_count, batch_size],
            outputs=txt2img_status,
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

        txt2img_png_info_img.change(
            fn=import_png_metadata,
            inputs=[
                txt2img_png_info_img,
                prompt,
                negative_prompt,
                steps,
                scheduler,
                guidance_scale,
                seed,
                width,
                height,
                txt2img_custom_model,
                lora_weights,
                lora_hf_id,
                custom_vae,
            ],
            outputs=[
                txt2img_png_info_img,
                prompt,
                negative_prompt,
                steps,
                scheduler,
                guidance_scale,
                seed,
                width,
                height,
                txt2img_custom_model,
                lora_weights,
                lora_hf_id,
                custom_vae,
            ],
        )

        # SharkEulerDiscrete doesn't work with img2img which hires_fix uses
        def set_compatible_schedulers(hires_fix_selected):
            if hires_fix_selected:
                return gr.Dropdown(
                    choices=scheduler_list_cpu_only,
                    value="DEISMultistep",
                )
            else:
                return gr.Dropdown(
                    choices=scheduler_list,
                    value="SharkEulerDiscrete",
                )

        use_hiresfix.change(
            fn=set_compatible_schedulers,
            inputs=[use_hiresfix],
            outputs=[scheduler],
            queue=False,
        )

        lora_weights.change(
            fn=lora_changed,
            inputs=[lora_weights],
            outputs=[lora_tags],
            queue=True,
        )
