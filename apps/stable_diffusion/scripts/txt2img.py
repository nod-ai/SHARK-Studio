import torch
import time
from apps.stable_diffusion.src import (
    args,
    Text2ImagePipeline,
    Text2ImageSDXLPipeline,
    get_schedulers,
    set_init_device_flags,
    utils,
    clear_all,
    save_output_img,
)


def main():
    if args.clear_all:
        clear_all()

    # TODO: prompt_embeds and text_embeds form base_model.json requires fixing
    dtype = torch.float32 if args.precision == "fp32" else torch.half
    cpu_scheduling = not args.scheduler.startswith("Shark")
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)
    scheduler_obj = schedulers[args.scheduler]
    seed = args.seed
    if args.height == 1024:
        assert (
            args.width == 1024
        ), "currently we support only 1024x1024 image size via SDXL"
        assert args.precision == "fp16", "currently we support fp16 for SDXL"
        # For SDXL we set max_length as 77.
        args.max_length = 77
        txt2img_obj = Text2ImageSDXLPipeline.from_pretrained(
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
            use_quantize=args.use_quantize,
            ondemand=args.ondemand,
        )
    else:
        assert (
            args.height <= 768 and args.width <= 768
        ), "height/width not in supported range"
        txt2img_obj = Text2ImagePipeline.from_pretrained(
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
            use_quantize=args.use_quantize,
            ondemand=args.ondemand,
        )

    seeds = utils.batch_seeds(seed, args.batch_count, args.repeatable_seeds)
    for current_batch in range(args.batch_count):
        start_time = time.time()
        generated_imgs = txt2img_obj.generate_images(
            args.prompts,
            args.negative_prompts,
            args.batch_size,
            args.height,
            args.width,
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
        text_output += (
            f"seed={seeds[current_batch]}, size={args.height}x{args.width}"
        )
        text_output += (
            f", batch size={args.batch_size}, max_length={args.max_length}"
        )
        # TODO: if using --batch_count=x txt2img_obj.log will output on each display every iteration infos from the start
        text_output += txt2img_obj.log
        text_output += f"\nTotal image generation time: {total_time:.4f}sec"

        save_output_img(generated_imgs[0], seed)
        print(text_output)


if __name__ == "__main__":
    main()
