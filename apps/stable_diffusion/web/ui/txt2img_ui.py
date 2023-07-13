import os
import torch
import time
import sys
import gradio as gr
from PIL import Image
import base64
from io import BytesIO
from fastapi.exceptions import HTTPException
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
    Text2ImagePipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
    save_output_img,
    prompt_examples,
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


def txt2img_inf(
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    custom_model: str,
    hf_model_id: str,
    custom_vae: str,
    precision: str,
    device: str,
    max_length: int,
    save_metadata_to_json: bool,
    save_metadata_to_png: bool,
    lora_weights: str,
    lora_hf_id: str,
    ondemand: bool,
    std_output: str = args.output_dir,
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
    if custom_model == "None":
        if not hf_model_id:
            return (
                None,
                "Please provide either custom model or huggingface model ID, "
                "both must not be empty",
            )
        if "civitai" in hf_model_id:
            args.ckpt_loc = hf_model_id
        else:
            args.hf_model_id = hf_model_id
    elif ".ckpt" in custom_model or ".safetensors" in custom_model:
        args.ckpt_loc = get_custom_model_pathfile(custom_model)
    else:
        args.hf_model_id = custom_model
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
        use_stencil=None,
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
        args.output_dir = str(std_output)
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
    seeds = []
    img_seed = utils.sanitize_seed(seed)
    text_output = ""
    for i in range(batch_count):
        if i > 0:
            img_seed = utils.sanitize_seed(-1)
        out_imgs = global_obj.get_sd_obj().generate_images(
            prompt,
            negative_prompt,
            batch_size,
            height,
            width,
            steps,
            guidance_scale,
            img_seed,
            args.max_length,
            dtype,
            args.use_base_vae,
            cpu_scheduling,
            args.max_embeddings_multiples,
        )
        seeds.append(img_seed)
        total_time = time.time() - start_time
        text_output = get_generation_text_info(seeds, device)
        text_output += "\n" + global_obj.get_sd_obj().log
        text_output += f"\nTotal image(s) generation time: {total_time:.4f}sec"

        if global_obj.get_sd_status() == SD_STATE_CANCEL:
            break
        else:
            save_output_img(out_imgs[0], img_seed)
            generated_imgs.extend(out_imgs)
            yield generated_imgs, text_output, status_label(
                "Text-to-Image", i + 1, batch_count, batch_size
            )

    return generated_imgs, text_output, ""


def encode_pil_to_base64(images):
    encoded_imgs = []
    for image in images:
        with BytesIO() as output_bytes:
            if args.output_img_format.lower() == "png":
                image.save(output_bytes, format="PNG")

            elif args.output_img_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG")
            else:
                raise HTTPException(
                    status_code=500, detail="Invalid image format"
                )
            bytes_data = output_bytes.getvalue()
            encoded_imgs.append(base64.b64encode(bytes_data))
    return encoded_imgs


# Text2Img Rest API.
def txt2img_api(
    InputData: dict,
):
    print(
        f'Prompt: {InputData["prompt"]}, '
        f'Negative Prompt: {InputData["negative_prompt"]}, '
        f'Seed: {InputData["seed"]}.'
    )
    res = txt2img_inf(
        InputData["prompt"],
        InputData["negative_prompt"],
        InputData["height"],
        InputData["width"],
        InputData["steps"],
        InputData["cfg_scale"],
        InputData["seed"],
        batch_count=1,
        batch_size=1,
        scheduler="EulerDiscrete",
        custom_model="None",
        hf_model_id=InputData["hf_model_id"]
        if "hf_model_id" in InputData.keys()
        else "stabilityai/stable-diffusion-2-1-base",
        custom_vae="None",
        precision="fp16",
        device=available_devices[0],
        max_length=64,
        save_metadata_to_json=False,
        save_metadata_to_png=False,
        lora_weights="None",
        lora_hf_id="",
        ondemand=False,
    )

    # Convert Generator to Subscriptable
    res = next(res)

    return {
        "images": encode_pil_to_base64(res[0]),
        "parameters": {},
        "info": res[1],
    }


with gr.Blocks(title="Text-to-Image") as txt2img_web:
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
                            # janky fix for overflowing text
                            t2i_model_info = (
                                str(get_custom_model_path())
                            ).replace("\\", "\n\\")
                            t2i_model_info = (
                                f"Custom Model Path: {t2i_model_info}"
                            )
                            txt2img_custom_model = gr.Dropdown(
                                label=f"Models",
                                info=t2i_model_info,
                                elem_id="custom_model",
                                value=os.path.basename(args.ckpt_loc)
                                if args.ckpt_loc
                                else "stabilityai/stable-diffusion-2-1-base",
                                choices=["None"]
                                + get_custom_model_files()
                                + predefined_models,
                            )
                            txt2img_hf_model_id = gr.Textbox(
                                elem_id="hf_model_id",
                                placeholder="Select 'None' in the dropdown "
                                "on the left and enter model ID here.",
                                value="",
                                label="HuggingFace Model ID or Civitai model "
                                "download URL.",
                                lines=3,
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
                                value=os.path.basename(args.custom_vae)
                                if args.custom_vae
                                else "None",
                                choices=["None"]
                                + get_custom_model_files("vae"),
                            )
                    with gr.Column(scale=1, min_width=170):
                        txt2img_png_info_img = gr.Image(
                            label="Import PNG info",
                            elem_id="txt2img_prompt_image",
                            type="pil",
                            tool="None",
                            visible=True,
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
                        # janky fix for overflowing text
                        t2i_lora_info = (
                            str(get_custom_model_path("lora"))
                        ).replace("\\", "\n\\")
                        t2i_lora_info = f"LoRA Path: {t2i_lora_info}"
                        lora_weights = gr.Dropdown(
                            label=f"Standalone LoRA Weights",
                            info=t2i_lora_info,
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
                with gr.Accordion(label="Advanced Options", open=False):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            elem_id="scheduler",
                            label="Scheduler",
                            value=args.scheduler,
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
                            384,
                            768,
                            value=args.height,
                            step=8,
                            label="Height",
                        )
                        width = gr.Slider(
                            384,
                            768,
                            value=args.width,
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
                        steps = gr.Slider(
                            1, 100, value=args.steps, step=1, label="Steps"
                        )
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
                        stable_diffusion = gr.Button("Generate Image(s)")

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
                        lines=2,
                        elem_id="std_output",
                        label="Image Save Location",
                        show_label=False,
                        interactive=True,
                        placeholder=f"Images will be saved at {get_generated_imgs_path()} by default.\nTo save the images in a custom directory, type the path to an already created directory to save the images in.",
                    )
                    txt2img_status = gr.Textbox(visible=False)
                with gr.Row():
                    txt2img_sendto_img2img = gr.Button(value="SendTo Img2Img")
                    txt2img_sendto_inpaint = gr.Button(value="SendTo Inpaint")
                    txt2img_sendto_outpaint = gr.Button(
                        value="SendTo Outpaint"
                    )
                    txt2img_sendto_upscaler = gr.Button(
                        value="SendTo Upscaler"
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
                txt2img_hf_model_id,
                custom_vae,
                precision,
                device,
                max_length,
                save_metadata_to_json,
                save_metadata_to_png,
                lora_weights,
                lora_hf_id,
                ondemand,
                std_output,
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
                txt2img_hf_model_id,
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
                txt2img_hf_model_id,
                lora_weights,
                lora_hf_id,
                custom_vae,
            ],
        )
