from

# save output images and the inputs corresponding to it.
def save_output_img(output_img, img_seed, extra_info=None):
    if extra_info is None:
        extra_info = {}
    generated_imgs_path = Path(
        get_generated_imgs_path(), get_generated_imgs_todays_subdir()
    )
    generated_imgs_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(generated_imgs_path, "imgs_details.csv")

    prompt_slice = re.sub("[^a-zA-Z0-9]", "_", args.prompts[0][:15])
    out_img_name = f"{dt.now().strftime('%H%M%S')}_{prompt_slice}_{img_seed}"

    img_model = args.hf_model_id
    if args.ckpt_loc:
        img_model = Path(os.path.basename(args.ckpt_loc)).stem

    img_vae = None
    if args.custom_vae:
        img_vae = Path(os.path.basename(args.custom_vae)).stem

    img_lora = None
    if args.use_lora:
        img_lora = Path(os.path.basename(args.use_lora)).stem

    if args.output_img_format == "jpg":
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.jpg")
        output_img.save(out_img_path, quality=95, subsampling=0)
    else:
        out_img_path = Path(generated_imgs_path, f"{out_img_name}.png")
        pngInfo = PngImagePlugin.PngInfo()

        if args.write_metadata_to_png:
            # Using a conditional expression caused problems, so setting a new
            # variable for now.
            if args.use_hiresfix:
                png_size_text = f"{args.hiresfix_width}x{args.hiresfix_height}"
            else:
                png_size_text = f"{args.width}x{args.height}"

            pngInfo.add_text(
                "parameters",
                f"{args.prompts[0]}"
                f"\nNegative prompt: {args.negative_prompts[0]}"
                f"\nSteps: {args.steps},"
                f"Sampler: {args.scheduler}, "
                f"CFG scale: {args.guidance_scale}, "
                f"Seed: {img_seed},"
                f"Size: {png_size_text}, "
                f"Model: {img_model}, "
                f"VAE: {img_vae}, "
                f"LoRA: {img_lora}",
            )

        output_img.save(out_img_path, "PNG", pnginfo=pngInfo)

        if args.output_img_format not in ["png", "jpg"]:
            print(
                f"[ERROR] Format {args.output_img_format} is not "
                f"supported yet. Image saved as png instead."
                f"Supported formats: png / jpg"
            )

    # To be as low-impact as possible to the existing CSV format, we append
    # "VAE" and "LORA" to the end. However, it does not fit the hierarchy of
    # importance for each data point. Something to consider.
    new_entry = {
        "VARIANT": img_model,
        "SCHEDULER": args.scheduler,
        "PROMPT": args.prompts[0],
        "NEG_PROMPT": args.negative_prompts[0],
        "SEED": img_seed,
        "CFG_SCALE": args.guidance_scale,
        "PRECISION": args.precision,
        "STEPS": args.steps,
        "HEIGHT": args.height
        if not args.use_hiresfix
        else args.hiresfix_height,
        "WIDTH": args.width if not args.use_hiresfix else args.hiresfix_width,
        "MAX_LENGTH": args.max_length,
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

    if args.save_metadata_to_json:
        del new_entry["OUTPUT"]
        json_path = Path(generated_imgs_path, f"{out_img_name}.json")
        with open(json_path, "w") as f:
            json.dump(new_entry, f, indent=4)


def get_generation_text_info(seeds, device):
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += (
        f"\nmodel_id={args.hf_model_id}, " f"ckpt_loc={args.ckpt_loc}"
    )
    text_output += f"\nscheduler={args.scheduler}, " f"device={device}"
    text_output += (
        f"\nsteps={args.steps}, "
        f"guidance_scale={args.guidance_scale}, "
        f"seed={seeds}"
    )
    text_output += (
        f"\nsize={args.height}x{args.width}, "
        if not args.use_hiresfix
        else f"\nsize={args.hiresfix_height}x{args.hiresfix_width}, "
    )
    text_output += (
        f"batch_count={args.batch_count}, "
        f"batch_size={args.batch_size}, "
        f"max_length={args.max_length}"
    )

    return text_output


# For stencil, the input image can be of any size, but we need to ensure that
# it conforms with our model constraints :-
#   Both width and height should be in the range of [128, 768] and multiple of 8.
# This utility function performs the transformation on the input image while
# also maintaining the aspect ratio before sending it to the stencil pipeline.
def resize_stencil(image: Image.Image, width, height):
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
    new_image = image.resize((n_width, n_height))
    return new_image, n_width, n_height

