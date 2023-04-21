import torch
import time
from PIL import Image
import transformers
from apps.stable_diffusion.src import (
    args,
    UpscalerPipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
    clear_all,
    save_output_img,
)


if __name__ == "__main__":
    if args.clear_all:
        clear_all()

    if args.img_path is None:
        print("Flag --img_path is required.")
        exit()

    # When the models get uploaded, it should be default to False.
    args.import_mlir = True

    cpu_scheduling = not args.scheduler.startswith("Shark")
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)

    scheduler_obj = schedulers[args.scheduler]
    image = (
        Image.open(args.img_path)
        .convert("RGB")
        .resize((args.height, args.width))
    )
    seed = utils.sanitize_seed(args.seed)
    # Adjust for height and width based on model

    upscaler_obj = UpscalerPipeline.from_pretrained(
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
        ddpm_scheduler=schedulers["DDPM"],
        ondemand=args.ondemand,
    )

    start_time = time.time()
    generated_imgs = upscaler_obj.generate_images(
        args.prompts,
        args.negative_prompts,
        image,
        args.batch_size,
        args.height,
        args.width,
        args.steps,
        args.noise_level,
        args.guidance_scale,
        seed,
        args.max_length,
        dtype,
        args.use_base_vae,
        cpu_scheduling,
    )
    total_time = time.time() - start_time
    text_output = f"prompt={args.prompts}"
    text_output += f"\nnegative prompt={args.negative_prompts}"
    text_output += f"\nmodel_id={args.hf_model_id}, ckpt_loc={args.ckpt_loc}"
    text_output += f"\nscheduler={args.scheduler}, device={args.device}"
    text_output += f"\nsteps={args.steps}, noise_level={args.noise_level}, guidance_scale={args.guidance_scale}, seed={seed}, size={args.height}x{args.width}"
    text_output += (
        f", batch size={args.batch_size}, max_length={args.max_length}"
    )
    text_output += upscaler_obj.log
    text_output += f"\nTotal image generation time: {total_time:.4f}sec"

    extra_info = {"NOISE LEVEL": args.noise_level}
    save_output_img(generated_imgs[0], seed, extra_info)
    print(text_output)
