import torch
import time
from PIL import Image
import transformers
from apps.stable_diffusion.src import (
    args,
    InpaintPipeline,
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
    if args.mask_path is None:
        print("Flag --mask_path is required.")
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
    mask_image = Image.open(args.mask_path)

    inpaint_obj = InpaintPipeline.from_pretrained(
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
        ondemand=args.ondemand,
    )

    seeds = utils.batch_seeds(seed, args.batch_count, args.repeatable_seeds)
    for current_batch in range(args.batch_count):
        start_time = time.time()
        generated_imgs = inpaint_obj.generate_images(
            args.prompts,
            args.negative_prompts,
            image,
            mask_image,
            args.batch_size,
            args.height,
            args.width,
            args.inpaint_full_res,
            args.inpaint_full_res_padding,
            args.steps,
            args.guidance_scale,
            seeds[current_batch],
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
        text_output += (
            f"\nsteps={args.steps}, guidance_scale={args.guidance_scale},"
        )
        text_output += f"seed={seed}, size={args.height}x{args.width}"
        text_output += (
            f", batch size={args.batch_size}, max_length={args.max_length}"
        )
        text_output += inpaint_obj.log
        text_output += f"\nTotal image generation time: {total_time:.4f}sec"

        save_output_img(generated_imgs[0], seed)
        print(text_output)


if __name__ == "__main__":
    main()
