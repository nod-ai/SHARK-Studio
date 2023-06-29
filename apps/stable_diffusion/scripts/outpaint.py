import torch
import time
from PIL import Image
import transformers
from apps.stable_diffusion.src import (
    args,
    OutpaintPipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
    clear_all,
    save_output_img,
)


def main():
    if args.clear_all:
        clear_all()

    if args.img_path is None:
        print("Flag --img_path is required.")
        exit()

    dtype = torch.float32 if args.precision == "fp32" else torch.half
    cpu_scheduling = not args.scheduler.startswith("Shark")
    set_init_device_flags()
    model_id = (
        args.hf_model_id
        if "inpaint" in args.hf_model_id
        else "stabilityai/stable-diffusion-2-inpainting"
    )
    schedulers = get_schedulers(model_id)
    scheduler_obj = schedulers[args.scheduler]
    seed = args.seed
    image = Image.open(args.img_path)

    outpaint_obj = OutpaintPipeline.from_pretrained(
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
        use_lora=args.use_lora,
        ondemand=args.ondemand,
    )

    for current_batch in range(args.batch_count):
        if current_batch > 0:
            seed = -1
        seed = utils.sanitize_seed(seed)

        start_time = time.time()
        generated_imgs = outpaint_obj.generate_images(
            args.prompts,
            args.negative_prompts,
            image,
            args.pixels,
            args.mask_blur,
            args.left,
            args.right,
            args.top,
            args.bottom,
            args.noise_q,
            args.color_variation,
            args.batch_size,
            args.height,
            args.width,
            args.steps,
            args.guidance_scale,
            seed,
            args.max_length,
            dtype,
            args.use_base_vae,
            cpu_scheduling,
            args.max_embeddings_multiples,
        )
        total_time = time.time() - start_time
        text_output = f"prompt={args.prompts}"
        text_output += f"\nnegative prompt={args.negative_prompts}"
        text_output += (
            f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
        )
        text_output += f"\nscheduler={args.scheduler}, device={args.device}"
        text_output += f"\nsteps={args.steps}, guidance_scale={args.guidance_scale}, seed={seed}, size={args.height}x{args.width}"
        text_output += (
            f", batch size={args.batch_size}, max_length={args.max_length}"
        )
        text_output += outpaint_obj.log
        text_output += f"\nTotal image generation time: {total_time:.4f}sec"

        # save this information as metadata of output generated image.
        directions = []
        if args.left:
            directions.append("left")
        if args.right:
            directions.append("right")
        if args.top:
            directions.append("up")
        if args.bottom:
            directions.append("down")
        extra_info = {
            "PIXELS": args.pixels,
            "MASK_BLUR": args.mask_blur,
            "DIRECTIONS": directions,
            "NOISE_Q": args.noise_q,
            "COLOR_VARIATION": args.color_variation,
        }
        save_output_img(generated_imgs[0], seed, extra_info)
        print(text_output)


if __name__ == "__main__":
    main()
