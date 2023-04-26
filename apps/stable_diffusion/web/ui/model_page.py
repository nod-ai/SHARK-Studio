from pathlib import Path
import os
import torch
import time
import sys
import gradio as gr
from PIL import Image
import base64
from io import BytesIO
import json
from shark.iree_utils._common import run_cmd
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

def get_civit_list(num_of_models = 10):
    cmd = f"curl https://civitai.com/api/v1/models\?limit\={num_of_models}\&nsfw\=false -H \"Content-Type: application/json\" -X GET"
    raw_json, error_msg = run_cmd(cmd)
    models = list(json.loads(raw_json).items())[0][1]
    safe_models = [safe_model for safe_model in models if not safe_model["nsfw"]]
    version_id = 0 # Currently just using the first version.
    first_version_models = []
    for model_iter in safe_models:
        # The modelVersion would only keep the version name.
        model_iter["modelVersions"][version_id]["modelName"] = model_iter["name"]
        first_version_models.append(model_iter["modelVersions"][version_id])
    return first_version_models

def get_image_from_model(model_json):
    model_id = model_json["modelId"]
    image_url = model_json["images"][0]["url"]
    save_img = f"curl {image_url} -o /tmp/{model_id}.jpeg"
    run_cmd(save_img)
    return f"/tmp/{model_id}.jpeg"

model_list = get_civit_list()
with gr.Blocks() as model_web:
    with gr.Column():
        for model in model_list:
            with gr.Row():
                img_path = get_image_from_model(model)
                model_img = Image.open(img_path)
                gr.Image(
                    value=model_img,
                    show_label=False,
                    interactive=False,
                    elem_id="top_logo",
                ).style(width=300, height=300)
                btn2 = gr.Button(f'Try {model["modelName"]}')
