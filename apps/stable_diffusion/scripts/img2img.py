import sys
import torch
import time
from PIL import Image
import transformers
from apps.stable_diffusion.src import (
    args,
    Image2ImagePipeline,
    StencilPipeline,
    resize_stencil,
    get_schedulers,
    set_init_device_flags,
    utils,
    clear_all,
    save_output_img,
)
from apps.stable_diffusion.src.utils import get_generation_text_info


def main():
    if args.clear_all:
        clear_all()

    if args.img_path is None:
        print("Flag --img_path is required.")
        exit()

    image = Image.open(args.img_path).convert("RGB")
    # When the models get uploaded, it should be default to False.
    args.import_mlir = True

    use_stencil = args.use_stencil
    if use_stencil:
        args.scheduler = "DDIM"
        args.hf_model_id = "runwayml/stable-diffusion-v1-5"
        image, args.width, args.height = resize_stencil(image)
    elif "Shark" in args.scheduler:
        print(
            f"Shark schedulers are not supported. Switching to EulerDiscrete scheduler"
        )
        args.scheduler = "EulerDiscrete"
    cpu_scheduling = not args.scheduler.startswith("Shark")
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)
    scheduler_obj = schedulers[args.scheduler]
    seed = utils.sanitize_seed(args.seed)
    # Adjust for height and width based on model

    if use_stencil:
        img2img_obj = StencilPipeline.from_pretrained(
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
            use_stencil=use_stencil,
            debug=args.import_debug if args.import_mlir else False,
            use_lora=args.use_lora,
            ondemand=args.ondemand,
        )
    else:
        img2img_obj = Image2ImagePipeline.from_pretrained(
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

    start_time = time.time()
    generated_imgs = img2img_obj.generate_images(
        args.prompts,
        args.negative_prompts,
        image,
        args.batch_size,
        args.height,
        args.width,
        args.steps,
        args.strength,
        args.guidance_scale,
        seed,
        args.max_length,
        dtype,
        args.use_base_vae,
        cpu_scheduling,
        args.max_embeddings_multiples,
        use_stencil=use_stencil,
    )
    total_time = time.time() - start_time
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
    text_output += f"\nscheduler={args.scheduler}, device={args.device}"
    text_output += f"\nsteps={args.steps}, strength={args.strength}, guidance_scale={args.guidance_scale}, seed={seed}, size={args.height}x{args.width}"
    text_output += (
        f", batch size={args.batch_size}, max_length={args.max_length}"
    )
    text_output += img2img_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    extra_info = {"STRENGTH": args.strength}
    save_output_img(generated_imgs[0], seed, extra_info)
    print(text_output)


if __name__ == "__main__":
    main()
