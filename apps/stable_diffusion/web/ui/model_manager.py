import os
import gradio as gr
import requests
from PIL import Image
from shark.iree_utils._common import run_cmd


def get_hf_list(limit=20):
    path = "https://huggingface.co/api/models"
    params = {
        "search": "stable-diffusion",
        "sort": "downloads",
        "direction": "-1",
        "limit": {limit},
        "full": "true",
    }
    response = requests.get(path, params=params)
    return response.json()


def get_civit_list(num_of_models=50):
    path = f"https://civitai.com/api/v1/models?limit={num_of_models}&types=Checkpoint"
    headers = {"Content-Type": "application/json"}
    raw_json = requests.get(path, headers=headers).json()
    models = list(raw_json.items())[0][1]
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


hf_model_list = get_hf_list()
civit_model_list = get_civit_list()


with gr.Blocks() as model_web:
    model_source = gr.Radio(
        choices=["Hugging Face", "Civitai"],
        type="index",
        value="Hugging Face",
        label="Model Source",
    )
    with gr.Column(visible=True) as hf_block:
        for model in hf_model_list:
            with gr.Row():
                model_url = gr.Textbox(
                    label="Model ID:",
                    value=model["modelId"],
                    lines=1,
                    interactive=False,
                )
                model_info = gr.Textbox(
                    value=f'Download Count: {model["downloads"]}{os.linesep}Favorite Count: {model["likes"]}',
                    lines=2,
                    show_label=False,
                    interactive=False,
                )
    with gr.Column(visible=False) as civit_block:
        for model in civit_model_list:
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

    def update_model_list(model_source):
        if model_source:
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)

    model_source.change(
        fn=update_model_list,
        inputs=model_source,
        outputs=[hf_block, civit_block],
    )
