import os
import gradio as gr
from PIL import Image
import json
from shark.iree_utils._common import run_cmd


def get_civit_list(num_of_models=100):
    cmd = f'curl https://civitai.com/api/v1/models\?limit\={num_of_models}\&types\=Checkpoint -H "Content-Type: application/json" -X GET'
    raw_json, error_msg = run_cmd(cmd)
    models = list(json.loads(raw_json).items())[0][1]
    safe_models = [
        safe_model for safe_model in models if not safe_model["nsfw"]
    ]
    version_id = 0  # Currently just using the first version.
    safe_models = [
        safe_model
        for safe_model in safe_models
        if safe_model["modelVersions"][version_id]["files"][0]["metadata"][
            "format"
        ]
        == "SafeTensor"
    ]
    first_version_models = []
    for model_iter in safe_models:
        # The modelVersion would only keep the version name.
        if (
            model_iter["modelVersions"][version_id]["images"][0]["nsfw"]
            != "None"
        ):
            continue
        model_iter["modelVersions"][version_id]["modelName"] = model_iter[
            "name"
        ]
        model_iter["modelVersions"][version_id]["rating"] = model_iter[
            "stats"
        ]["rating"]
        model_iter["modelVersions"][version_id]["favoriteCount"] = model_iter[
            "stats"
        ]["favoriteCount"]
        model_iter["modelVersions"][version_id]["downloadCount"] = model_iter[
            "stats"
        ]["downloadCount"]
        first_version_models.append(model_iter["modelVersions"][version_id])
    return first_version_models


def get_image_from_model(model_json):
    model_id = model_json["modelId"]
    path = None
    for img_info in model_json["images"]:
        if img_info["nsfw"] == "None":
            image_url = model_json["images"][0]["url"]
            save_img = f"curl {image_url} -o /tmp/{model_id}.jpeg"
            run_cmd(save_img)
            path = f"/tmp/{model_id}.jpeg"
            break
    return path


model_list = get_civit_list()
with gr.Blocks() as model_web:
    for model in model_list:
        with gr.Row():
            img_path = get_image_from_model(model)
            if img_path is None:
                continue
            model_img = Image.open(img_path)
            gr.Image(
                value=model_img,
                show_label=False,
                interactive=False,
                elem_id="top_logo",
            ).style(width=300, height=300)
            with gr.Column():
                gr.Textbox(
                    label=f'{model["modelName"]}',
                    value=f'Rating: {model["rating"]}{os.linesep}Favorite Count: {model["favoriteCount"]}{os.linesep}Download Count: {model["downloadCount"]}{os.linesep}File Format: {model["files"][0]["metadata"]["format"]}',
                    lines=4,
                )
                gr.Textbox(
                    label="Download URL:",
                    value=f'{model["files"][0]["downloadUrl"]}',
                    lines=1,
                )
