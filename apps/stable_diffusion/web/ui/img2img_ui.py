import os
import torch
import time
import gradio as gr
import PIL
from math import ceil
from PIL import Image

from apps.stable_diffusion.web.ui.utils import (
    available_devices,
    nodlogo_loc,
    get_custom_model_path,
    get_custom_model_files,
    scheduler_list_cpu_only,
    predefined_models,
    cancel_sd,
)
from apps.stable_diffusion.web.ui.common_ui_events import lora_changed
from apps.stable_diffusion.src import (
    args,
    Image2ImagePipeline,
    StencilPipeline,
    resize_stencil,
    get_schedulers,
    set_init_device_flags,
    utils,
    save_output_img,
)
from apps.stable_diffusion.src.utils import (
    get_generated_imgs_path,
    get_generation_text_info,
    resampler_list,
)
from apps.stable_diffusion.src.utils.stencils import (
    CannyDetector,
    OpenposeDetector,
    ZoeDetector,
)
from apps.stable_diffusion.web.utils.common_label_calc import status_label
import numpy as np


# set initial values of iree_vulkan_target_triple, use_tuned and import_mlir.
init_iree_vulkan_target_triple = args.iree_vulkan_target_triple
init_use_tuned = args.use_tuned
init_import_mlir = args.import_mlir


# Exposed to UI.
def img2img_inf(
    prompt: str,
    negative_prompt: str,
    image_dict,
    height: int,
    width: int,
    steps: int,
    strength: float,
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
    resample_type: str,
    control_mode: str,
    stencils: list,
    images: list,
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
    args.strength = strength
    args.scheduler = scheduler
    args.img_path = "not none"
    args.ondemand = ondemand

    for i, stencil in enumerate(stencils):
        if images[i] is None and stencil is not None:
            return
        if images[i] is not None:
            if isinstance(images[i], dict):
                images[i] = images[i]["composite"]
            images[i] = images[i].convert("RGB")

    if image_dict is None and images[0] is None:
        return None, "An Initial Image is required"
    if isinstance(image_dict, PIL.Image.Image):
        image = image_dict.convert("RGB")
    elif image_dict:
        image = image_dict["image"].convert("RGB")
    else:
        # TODO: enable t2i + controlnets
        image = None

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

    args.use_lora = get_custom_vae_or_lora_weights(
        lora_weights, lora_hf_id, "lora"
    )

    args.save_metadata_to_json = save_metadata_to_json
    args.write_metadata_to_png = save_metadata_to_png

    stencil_count = 0
    for stencil in stencils:
        if stencil is not None:
            stencil_count += 1
    if stencil_count > 0:
        args.hf_model_id = "runwayml/stable-diffusion-v1-5"
        image, width, height = resize_stencil(image)
    elif "Shark" in args.scheduler:
        print(
            f"Shark schedulers are not supported. Switching to EulerDiscrete "
            f"scheduler"
        )
        args.scheduler = "EulerDiscrete"
    cpu_scheduling = not args.scheduler.startswith("Shark")
    args.precision = precision
    dtype = torch.float32 if precision == "fp32" else torch.half
    new_config_obj = Config(
        "img2img",
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
        stencils=stencils,
        ondemand=ondemand,
    )
    if (
        not global_obj.get_sd_obj()
        or global_obj.get_cfg_obj() != new_config_obj
    ):
        global_obj.clear_cache()
        global_obj.set_cfg_obj(new_config_obj)
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
            else "stabilityai/stable-diffusion-2-1-base"
        )
        global_obj.set_schedulers(get_schedulers(model_id))
        scheduler_obj = global_obj.get_scheduler(args.scheduler)

        if stencil_count > 0:
            args.use_tuned = False
            global_obj.set_sd_obj(
                StencilPipeline.from_pretrained(
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
                    stencils=stencils,
                    debug=args.import_debug if args.import_mlir else False,
                    use_lora=args.use_lora,
                    ondemand=args.ondemand,
                )
            )
        else:
            global_obj.set_sd_obj(
                Image2ImagePipeline.from_pretrained(
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
                    debug=args.import_debug if args.import_mlir else False,
                    use_lora=args.use_lora,
                    ondemand=args.ondemand,
                )
            )

    global_obj.set_sd_scheduler(args.scheduler)

    start_time = time.time()
    global_obj.get_sd_obj().log = ""
    generated_imgs = []
    extra_info = {"STRENGTH": strength}
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
            batch_size,
            height,
            width,
            ceil(steps / strength),
            strength,
            guidance_scale,
            seeds[current_batch],
            args.max_length,
            dtype,
            args.use_base_vae,
            cpu_scheduling,
            args.max_embeddings_multiples,
            stencils,
            images,
            resample_type=resample_type,
            control_mode=control_mode,
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
            save_output_img(
                out_imgs[0],
                seeds[current_batch],
                extra_info,
            )
            generated_imgs.extend(out_imgs)
            yield generated_imgs, text_output, status_label(
                "Image-to-Image", current_batch + 1, batch_count, batch_size
            ), stencils, images

    return generated_imgs, text_output, "", stencils, images


with gr.Blocks(title="Image-to-Image") as img2img_web:
    # Stencils
    # TODO: Add more stencils here
    STENCIL_COUNT = 2
    stencils = gr.State([None] * STENCIL_COUNT)
    images = gr.State([None] * STENCIL_COUNT)
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
                    # janky fix for overflowing text
                    i2i_model_info = (
                        f"Custom Model Path: {str(get_custom_model_path())}"
                    )
                    img2img_custom_model = gr.Dropdown(
                        label=f"Models",
                        info="Select, or enter HuggingFace Model ID or Civitai model download URL",
                        elem_id="custom_model",
                        value=os.path.basename(args.ckpt_loc)
                        if args.ckpt_loc
                        else "stabilityai/stable-diffusion-2-1-base",
                        choices=get_custom_model_files() + predefined_models,
                        allow_custom_value=True,
                        scale=2,
                    )
                    # janky fix for overflowing text
                    i2i_vae_info = (str(get_custom_model_path("vae"))).replace(
                        "\\", "\n\\"
                    )
                    i2i_vae_info = f"VAE Path: {i2i_vae_info}"
                    custom_vae = gr.Dropdown(
                        label=f"Custom VAE Models",
                        info=i2i_vae_info,
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
                # TODO: make this import image prompt info if it exists
                img2img_init_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    height=512,
                    interactive=True,
                )

                with gr.Accordion(label="Multistencil Options", open=False):
                    choices = [
                        "None",
                        "canny",
                        "openpose",
                        "scribble",
                        "zoedepth",
                    ]

                    def cnet_preview(
                        model, input_image, index, stencils, images
                    ):
                        images[index] = input_image
                        stencils[index] = model
                        match model:
                            case "canny":
                                canny = CannyDetector()
                                result = canny(
                                    np.array(input_image["composite"]),
                                    100,
                                    200,
                                )
                                return (
                                    Image.fromarray(result),
                                    stencils,
                                    images,
                                )
                            case "openpose":
                                openpose = OpenposeDetector()
                                result = openpose(
                                    np.array(input_image["composite"])
                                )
                                # TODO: This is just an empty canvas, need to draw the candidates (which are in result[1])
                                return (
                                    Image.fromarray(result[0]),
                                    stencils,
                                    images,
                                )
                            case "zoedepth":
                                zoedepth = ZoeDetector()
                                result = zoedepth(
                                    np.array(input_image["composite"])
                                )
                                return (
                                    Image.fromarray(result),
                                    stencils,
                                    images,
                                )
                            case "scribble":
                                result = input_image["composite"].convert("L")
                                return (result, stencils, images)
                            case _:
                                return (None, stencils, images)

                    def create_canvas(width, height):
                        data = (
                            np.zeros(
                                shape=(height, width, 3),
                                dtype=np.uint8,
                            )
                            + 255
                        )
                        return data

                    def update_cn_input(model, width, height):
                        if model == "scribble":
                            return [
                                gr.ImageEditor(
                                    visible=True,
                                    image_mode="RGB",
                                    interactive=True,
                                    show_label=False,
                                    type="pil",
                                    value=create_canvas(width, height),
                                    crop_size=(width, height),
                                ),
                                gr.Image(
                                    visible=True,
                                    show_label=False,
                                    interactive=False,
                                ),
                                gr.Slider(visible=True),
                                gr.Slider(visible=True),
                                gr.Button(visible=True),
                            ]
                        else:
                            return [
                                gr.ImageEditor(
                                    visible=True,
                                    image_mode="RGB",
                                    type="pil",
                                    interactive=True,
                                    value=None,
                                ),
                                gr.Image(
                                    visible=True,
                                    show_label=False,
                                    interactive=True,
                                ),
                                gr.Slider(visible=False),
                                gr.Slider(visible=False),
                                gr.Button(visible=False),
                            ]

                    with gr.Row():
                        with gr.Column():
                            cnet_1 = gr.Button(
                                value="Generate controlnet input"
                            )
                            cnet_1_model = gr.Dropdown(
                                label="Controlnet 1",
                                value="None",
                                choices=choices,
                            )
                            canvas_width = gr.Slider(
                                label="Canvas Width",
                                minimum=256,
                                maximum=1024,
                                value=512,
                                step=1,
                                visible=False,
                            )
                            canvas_height = gr.Slider(
                                label="Canvas Height",
                                minimum=256,
                                maximum=1024,
                                value=512,
                                step=1,
                                visible=False,
                            )
                            make_canvas = gr.Button(
                                value="Make Canvas!",
                                visible=False,
                            )
                        cnet_1_image = gr.ImageEditor(
                            visible=False,
                            image_mode="RGB",
                            interactive=True,
                            show_label=False,
                            type="pil",
                        )
                        cnet_1_output = gr.Image(
                            visible=True, show_label=False
                        )
                        cnet_1_model.input(
                            update_cn_input,
                            [cnet_1_model, canvas_width, canvas_height],
                            [
                                cnet_1_image,
                                cnet_1_output,
                                canvas_width,
                                canvas_height,
                                make_canvas,
                            ],
                        )
                        make_canvas.click(
                            update_cn_input,
                            [cnet_1_model, canvas_width, canvas_height],
                            [
                                cnet_1_image,
                                cnet_1_output,
                                canvas_width,
                                canvas_height,
                                make_canvas,
                            ],
                        )
                        cnet_1.click(
                            fn=(
                                lambda a, b, s, i: cnet_preview(a, b, 0, s, i)
                            ),
                            inputs=[
                                cnet_1_model,
                                cnet_1_image,
                                stencils,
                                images,
                            ],
                            outputs=[cnet_1_output, stencils, images],
                        )
                    with gr.Row():
                        with gr.Column():
                            cnet_2 = gr.Button(
                                value="Generate controlnet input"
                            )
                            cnet_2_model = gr.Dropdown(
                                label="Controlnet 2",
                                value="None",
                                choices=choices,
                            )
                            canvas_width = gr.Slider(
                                label="Canvas Width",
                                minimum=256,
                                maximum=1024,
                                value=512,
                                step=1,
                                visible=False,
                            )
                            canvas_height = gr.Slider(
                                label="Canvas Height",
                                minimum=256,
                                maximum=1024,
                                value=512,
                                step=1,
                                visible=False,
                            )
                            make_canvas = gr.Button(
                                value="Make Canvas!",
                                visible=False,
                            )
                        cnet_2_image = gr.ImageEditor(
                            visible=False,
                            image_mode="RGB",
                            interactive=True,
                            show_label=False,
                            type="pil",
                        )
                        cnet_2_output = gr.Image(
                            visible=True, show_label=False
                        )
                        cnet_2_model.select(
                            update_cn_input,
                            [cnet_2_model, canvas_width, canvas_height],
                            [
                                cnet_2_image,
                                cnet_2_output,
                                canvas_width,
                                canvas_height,
                                make_canvas,
                            ],
                        )
                        make_canvas.click(
                            update_cn_input,
                            [cnet_2_model, canvas_width, canvas_height],
                            [
                                cnet_2_image,
                                cnet_2_output,
                                canvas_width,
                                canvas_height,
                                make_canvas,
                            ],
                        )
                        cnet_2.click(
                            fn=(
                                lambda a, b, s, i: cnet_preview(a, b, 1, s, i)
                            ),
                            inputs=[
                                cnet_2_model,
                                cnet_2_image,
                                stencils,
                                images,
                            ],
                            outputs=[cnet_2_output, stencils, images],
                        )
                    control_mode = gr.Radio(
                        choices=["Prompt", "Balanced", "Controlnet"],
                        value="Balanced",
                        label="Control Mode",
                    )

                with gr.Accordion(label="LoRA Options", open=False):
                    with gr.Row():
                        # janky fix for overflowing text
                        i2i_lora_info = (
                            str(get_custom_model_path("lora"))
                        ).replace("\\", "\n\\")
                        i2i_lora_info = f"LoRA Path: {i2i_lora_info}"
                        lora_weights = gr.Dropdown(
                            allow_custom_value=True,
                            label=f"Standalone LoRA Weights",
                            info=i2i_lora_info,
                            elem_id="lora_weights",
                            value="None",
                            choices=["None"] + get_custom_model_files("lora"),
                        )
                        lora_hf_id = gr.Textbox(
                            elem_id="lora_hf_id",
                            placeholder="Select 'None' in the Standalone LoRA "
                            "weights dropdown on the left if you want to use "
                            "a standalone HuggingFace model ID for LoRA here "
                            "e.g: sayakpaul/sd-model-finetuned-lora-t4",
                            value="",
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
                            strength = gr.Slider(
                                0,
                                1,
                                value=args.strength,
                                step=0.01,
                                label="Denoising Strength",
                            )
                            resample_type = gr.Dropdown(
                                value=args.resample_type,
                                choices=resampler_list,
                                label="Resample Type",
                                allow_custom_value=True,
                            )
                        ondemand = gr.Checkbox(
                            value=args.ondemand,
                            label="Low VRAM",
                            interactive=True,
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
                    img2img_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                        columns=2,
                        object_fit="contain",
                    )
                    std_output = gr.Textbox(
                        value=f"{i2i_model_info}\n"
                        f"Images will be saved at "
                        f"{get_generated_imgs_path()}",
                        lines=2,
                        elem_id="std_output",
                        show_label=False,
                    )
                    img2img_status = gr.Textbox(visible=False)
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
                    img2img_sendto_inpaint = gr.Button(value="SendTo Inpaint")
                    img2img_sendto_outpaint = gr.Button(
                        value="SendTo Outpaint"
                    )
                    img2img_sendto_upscaler = gr.Button(
                        value="SendTo Upscaler"
                    )

        kwargs = dict(
            fn=img2img_inf,
            inputs=[
                prompt,
                negative_prompt,
                img2img_init_image,
                height,
                width,
                steps,
                strength,
                guidance_scale,
                seed,
                batch_count,
                batch_size,
                scheduler,
                img2img_custom_model,
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
                resample_type,
                control_mode,
                stencils,
                images,
            ],
            outputs=[
                img2img_gallery,
                std_output,
                img2img_status,
                stencils,
                images,
            ],
            show_progress="minimal" if args.progress_bar else "none",
        )

        status_kwargs = dict(
            fn=lambda bc, bs: status_label("Image-to-Image", 0, bc, bs),
            inputs=[batch_count, batch_size],
            outputs=img2img_status,
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
