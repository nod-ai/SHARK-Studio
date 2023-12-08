import os
import sys
from PIL import Image
from pathlib import Path


# save output images and the inputs corresponding to it.
def save_output_img(output_img, img_seed, extra_info=None):
    if extra_info is None:
        extra_info = {}
    generated_imgs_path = Path(
        get_generated_imgs_path(), get_generated_imgs_todays_subdir()
    )
    generated_imgs_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(generated_imgs_path, "imgs_details.csv")

    prompt_slice = re.sub("[^a-zA-Z0-9]", "_", cmd_opts.prompts[0][:15])
    out_img_name = f"{dt.now().strftime('%H%M%S')}_{prompt_slice}_{img_seed}"

    img_model = cmd_opts.hf_model_id
    if cmd_opts.ckpt_loc:
        img_model = Path(os.path.basename(cmd_opts.ckpt_loc)).stem

    img_vae = None
    if cmd_opts.custom_vae:
        img_vae = Path(os.path.basename(cmd_opts.custom_vae)).stem

    img_lora = None
    if cmd_opts.use_lora:
        img_lora = Path(os.path.basename(cmd_opts.use_lora)).stem

    if cmd_opts.output_img_format == "jpg":
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.jpg")
        output_img.save(out_img_path, quality=95, subsampling=0)
    else:
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.png")
        pngInfo = PngImagePlugin.PngInfo()

        if cmd_opts.write_metadata_to_png:
            # Using a conditional expression caused problems, so setting a new
            # variable for now.
            if cmd_opts.use_hiresfix:
                png_size_text = (
                    f"{cmd_opts.hiresfix_width}x{cmd_opts.hiresfix_height}"
                )
            else:
                png_size_text = f"{cmd_opts.width}x{cmd_opts.height}"

            pngInfo.add_text(
                "parameters",
                f"{cmd_opts.prompts[0]}"
                f"\nNegative prompt: {cmd_opts.negative_prompts[0]}"
                f"\nSteps: {cmd_opts.steps},"
                f"Sampler: {cmd_opts.scheduler}, "
                f"CFG scale: {cmd_opts.guidance_scale}, "
                f"Seed: {img_seed},"
                f"Size: {png_size_text}, "
                f"Model: {img_model}, "
                f"VAE: {img_vae}, "
                f"LoRA: {img_lora}",
            )

        output_img.save(out_img_path, "PNG", pnginfo=pngInfo)

        if cmd_opts.output_img_format not in ["png", "jpg"]:
            print(
                f"[ERROR] Format {cmd_opts.output_img_format} is not "
                f"supported yet. Image saved as png instead."
                f"Supported formats: png / jpg"
            )

    # To be as low-impact as possible to the existing CSV format, we append
    # "VAE" and "LORA" to the end. However, it does not fit the hierarchy of
    # importance for each data point. Something to consider.
    new_entry = {
        "VARIANT": img_model,
        "SCHEDULER": cmd_opts.scheduler,
        "PROMPT": cmd_opts.prompts[0],
        "NEG_PROMPT": cmd_opts.negative_prompts[0],
        "SEED": img_seed,
        "CFG_SCALE": cmd_opts.guidance_scale,
        "PRECISION": cmd_opts.precision,
        "STEPS": cmd_opts.steps,
        "HEIGHT": cmd_opts.height
        if not cmd_opts.use_hiresfix
        else cmd_opts.hiresfix_height,
        "WIDTH": cmd_opts.width
        if not cmd_opts.use_hiresfix
        else cmd_opts.hiresfix_width,
        "MAX_LENGTH": cmd_opts.max_length,
        "OUTPUT": out_img_path,
        "VAE": img_vae,
        "LORA": img_lora,
    }

    new_entry.update(extra_info)

    csv_mode = "a" if os.path.isfile(csv_path) else "w"
    with open(csv_path, csv_mode, encoding="utf-8") as csv_obj:
        dictwriter_obj = DictWriter(csv_obj, fieldnames=list(new_entry.keys()))
        if csv_mode == "w":
            dictwriter_obj.writeheader()
        dictwriter_obj.writerow(new_entry)
        csv_obj.close()

    if cmd_opts.save_metadata_to_json:
        del new_entry["OUTPUT"]
        json_path = Path(generated_imgs_path, f"{out_img_name}.json")
        with open(json_path, "w") as f:
            json.dump(new_entry, f, indent=4)


resamplers = {
    "Lanczos": Image.Resampling.LANCZOS,
    "Nearest Neighbor": Image.Resampling.NEAREST,
    "Bilinear": Image.Resampling.BILINEAR,
    "Bicubic": Image.Resampling.BICUBIC,
    "Hamming": Image.Resampling.HAMMING,
    "Box": Image.Resampling.BOX,
}

resampler_list = resamplers.keys()


# For stencil, the input image can be of any size, but we need to ensure that
# it conforms with our model constraints :-
#   Both width and height should be in the range of [128, 768] and multiple of 8.
# This utility function performs the transformation on the input image while
# also maintaining the aspect ratio before sending it to the stencil pipeline.
def resize_stencil(image: Image.Image, width, height, resampler_type=None):
    aspect_ratio = width / height
    min_size = min(width, height)
    if min_size < 128:
        n_size = 128
        if width == min_size:
            width = n_size
            height = n_size / aspect_ratio
        else:
            height = n_size
            width = n_size * aspect_ratio
    width = int(width)
    height = int(height)
    n_width = width // 8
    n_height = height // 8
    n_width *= 8
    n_height *= 8

    min_size = min(width, height)
    if min_size > 768:
        n_size = 768
        if width == min_size:
            height = n_size
            width = n_size * aspect_ratio
        else:
            width = n_size
            height = n_size / aspect_ratio
    width = int(width)
    height = int(height)
    n_width = width // 8
    n_height = height // 8
    n_width *= 8
    n_height *= 8
    if resampler_type in resamplers:
        resampler = resamplers[resampler_type]
    else:
        resampler = resamplers["Nearest Neighbor"]
    new_image = image.resize((n_width, n_height), resampler=resampler)
    return new_image, n_width, n_height
