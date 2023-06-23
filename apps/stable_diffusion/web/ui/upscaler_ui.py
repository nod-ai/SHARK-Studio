import os
import torch
import time
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
    scheduler_list_cpu_only,
    predefined_upscaler_models,
    cancel_sd,
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
    if custom_model == "None":
        if not hf_model_id:
            return (
                None,
                "Please provide either custom model or huggingface model ID, both must not be empty",
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
        use_stencil=None,
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
    seeds = []
    img_seed = utils.sanitize_seed(seed)
    extra_info = {"NOISE LEVEL": noise_level}
    for current_batch in range(batch_count):
        if current_batch > 0:
            img_seed = utils.sanitize_seed(-1)
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
                    img_seed,
                    args.max_length,
                    dtype,
                    args.use_base_vae,
                    cpu_scheduling,
                )
                if global_obj.get_sd_status() == SD_STATE_CANCEL:
                    break
                else:
                    high_res_img.paste(upscaled_image[0], (j * 4, i * 4))

            if global_obj.get_sd_status() == SD_STATE_CANCEL:
                break

        if global_obj.get_sd_status() == SD_STATE_CANCEL:
            break
        else:
            save_output_img(high_res_img, img_seed, extra_info)
            generated_imgs.append(high_res_img)
            seeds.append(img_seed)
            global_obj.get_sd_obj().log += "\n"
            yield generated_imgs, global_obj.get_sd_obj().log, status_label(
                "Upscaler", current_batch + 1, batch_count, batch_size
            )

    total_time = time.time() - start_time
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
    text_output += f"\nscheduler={args.scheduler}, device={device}"
    text_output += f"\nsteps={steps}, noise_level={noise_level}, guidance_scale={guidance_scale}, seed={seeds}"
    text_output += f"\nsize={height}x{width}, batch_count={batch_count}, batch_size={batch_size}, max_length={args.max_length}"
    text_output += global_obj.get_sd_obj().log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    yield generated_imgs, text_output, ""


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";", 1)[1].split(",", 1)[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        print(err)
        raise HTTPException(status_code=500, detail="Invalid encoded image")


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


# Upscaler Rest API.
def upscaler_api(
    InputData: dict,
):
    print(
        f'Prompt: {InputData["prompt"]}, Negative Prompt: {InputData["negative_prompt"]}, Seed: {InputData["seed"]}'
    )
    init_image = decode_base64_to_image(InputData["init_images"][0])
    res = upscaler_inf(
        InputData["prompt"],
        InputData["negative_prompt"],
        init_image,
        InputData["height"],
        InputData["width"],
        InputData["steps"],
        InputData["noise_level"],
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
    # Converts generator type to subscriptable
    res = next(res)

    return {
        "images": encode_pil_to_base64(res[0]),
        "parameters": {},
        "info": res[1],
    }


with gr.Blocks(title="Upscaler") as upscaler_web:
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
                    upscaler_custom_model = gr.Dropdown(
                        label=f"Models (Custom Model path: {get_custom_model_path()})",
                        elem_id="custom_model",
                        value=os.path.basename(args.ckpt_loc)
                        if args.ckpt_loc
                        else "stabilityai/stable-diffusion-x4-upscaler",
                        choices=["None"]
                        + get_custom_model_files(
                            custom_checkpoint_type="upscaler"
                        )
                        + predefined_upscaler_models,
                    )
                    upscaler_hf_model_id = gr.Textbox(
                        elem_id="hf_model_id",
                        placeholder="Select 'None' in the Models dropdown on the left and enter model ID here e.g: SG161222/Realistic_Vision_V1.3, https://civitai.com/api/download/models/15236",
                        value="",
                        label="HuggingFace Model ID or Civitai model download URL",
                        lines=3,
                    )
                    custom_vae = gr.Dropdown(
                        label=f"Custom Vae Models (Path: {get_custom_model_path('vae')})",
                        elem_id="custom_model",
                        value=os.path.basename(args.custom_vae)
                        if args.custom_vae
                        else "None",
                        choices=["None"] + get_custom_model_files("vae"),
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

                upscaler_init_image = gr.Image(
                    label="Input Image", type="pil"
                ).style(height=300)

                with gr.Accordion(label="LoRA Options", open=False):
                    with gr.Row():
                        lora_weights = gr.Dropdown(
                            label=f"Standlone LoRA weights (Path: {get_custom_model_path('lora')})",
                            elem_id="lora_weights",
                            value="None",
                            choices=["None"] + get_custom_model_files("lora"),
                        )
                        lora_hf_id = gr.Textbox(
                            elem_id="lora_hf_id",
                            placeholder="Select 'None' in the Standlone LoRA weights dropdown on the left if you want to use a standalone HuggingFace model ID for LoRA here e.g: sayakpaul/sd-model-finetuned-lora-t4",
                            value="",
                            label="HuggingFace Model ID",
                            lines=3,
                        )
                with gr.Accordion(label="Advanced Options", open=False):
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            elem_id="scheduler",
                            label="Scheduler",
                            value="DDIM",
                            choices=scheduler_list_cpu_only,
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
                        batch_size = gr.Slider(
                            1,
                            4,
                            value=args.batch_size,
                            step=1,
                            label="Batch Size",
                            interactive=False,
                            visible=False,
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

            with gr.Column(scale=1, min_width=600):
                with gr.Group():
                    upscaler_gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(columns=[2], object_fit="contain")
                    std_output = gr.Textbox(
                        value=f"Images will be saved at {get_generated_imgs_path()}",
                        lines=1,
                        elem_id="std_output",
                        show_label=False,
                    )
                    upscaler_status = gr.Textbox(visible=False)

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
                upscaler_hf_model_id,
                custom_vae,
                precision,
                device,
                max_length,
                save_metadata_to_json,
                save_metadata_to_png,
                lora_weights,
                lora_hf_id,
                ondemand,
            ],
            outputs=[upscaler_gallery, std_output, upscaler_status],
            show_progress=args.progress_bar,
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
