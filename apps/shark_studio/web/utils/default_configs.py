default_sd_config = r"""{
  "prompt": [
    "a photo taken of the front of a super-car drifting on a road near mountains at high speeds with smoke coming off the tires, front angle, front point of view, trees in the mountains of the background, ((sharp focus))"
  ],
  "negative_prompt": [
    "watermark, signature, logo, text, lowres, ((monochrome, grayscale)), blurry, ugly, blur, oversaturated, cropped"
  ],
  "sd_init_image": [null],
  "height": 512,
  "width": 512,
  "steps": 50,
  "strength": 0.8,
  "guidance_scale": 7.5,
  "seed": "-1",
  "batch_count": 1,
  "batch_size": 1,
  "scheduler": "EulerDiscrete",
  "base_model_id": "stabilityai/stable-diffusion-2-1-base",
  "custom_weights": null,
  "custom_vae": null,
  "precision": "fp16",
  "device": "",
  "target_triple": "",
  "ondemand": false,
  "compiled_pipeline": false,
  "resample_type": "Nearest Neighbor",
  "controlnets": {},
  "embeddings": {}
}"""

sdxl_30steps = r"""{
  "prompt": [
    "a cat under the snow with blue eyes, covered by snow, cinematic style, medium shot, professional photo, animal"
  ],
  "negative_prompt": [
    "watermark, signature, logo, text, lowres, ((monochrome, grayscale)), blurry, ugly, blur, oversaturated, cropped"
  ],
  "sd_init_image": [null],
  "height": 1024,
  "width": 1024,
  "steps": 30,
  "strength": 0.8,
  "guidance_scale": 7.5,
  "seed": "-1",
  "batch_count": 1,
  "batch_size": 1,
  "scheduler": "EulerDiscrete",
  "base_model_id": "stabilityai/stable-diffusion-xl-base-1.0",
  "custom_weights": null,
  "custom_vae": null,
  "precision": "fp16",
  "device": "",
  "target_triple": "",
  "ondemand": false,
  "compiled_pipeline": true,
  "resample_type": "Nearest Neighbor",
  "controlnets": {},
  "embeddings": {}
}"""

sdxl_turbo = r"""{
  "prompt": [
    "A cat wearing a hat that says 'TURBO' on it. The cat is sitting on a skateboard."
  ],
  "negative_prompt": [
    ""
  ],
  "sd_init_image": [null],
  "height": 512,
  "width": 512,
  "steps": 2,
  "strength": 0.8,
  "guidance_scale": 0,
  "seed": "-1",
  "batch_count": 1,
  "batch_size": 1,
  "scheduler": "EulerAncestralDiscrete",
  "base_model_id": "stabilityai/sdxl-turbo",
  "custom_weights": null,
  "custom_vae": null,
  "precision": "fp16",
  "device": "",
  "target_triple": "",
  "ondemand": false,
  "compiled_pipeline": true,
  "resample_type": "Nearest Neighbor",
  "controlnets": {},
  "embeddings": {}
}"""

default_sd_configs = {
    "default_sd_config.json": default_sd_config,
    "sdxl-30steps.json": sdxl_30steps,
    "sdxl-turbo.json": sdxl_turbo,
}