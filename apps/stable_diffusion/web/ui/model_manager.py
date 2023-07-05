import os
import gradio as gr
import requests
from io import BytesIO
from PIL import Image


def get_hf_list(num_of_models=20):
    path = "https://huggingface.co/api/models"
    params = {
        "search": "stable-diffusion",
        "sort": "downloads",
        "direction": "-1",
        "limit": {num_of_models},
        "full": "true",
    }
    response = requests.get(path, params=params)
    return response.json()


def get_civit_list(num_of_models=50):
    path = (
        f"https://civitai.com/api/v1/models?limit="
        f"{num_of_models}&types=Checkpoint"
    )
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
    image = None
    for img_info in model_json["images"]:
        if img_info["nsfw"] == "None":
            image_url = model_json["images"][0]["url"]
            response = requests.get(image_url)
            image = BytesIO(response.content)
            break
    return image


with gr.Blocks() as model_web:
    with gr.Row():
        model_source = gr.Radio(
            value=None,
            choices=["Hugging Face", "Civitai"],
            type="value",
            label="Model Source",
        )
        model_number = gr.Slider(
            1,
            100,
            value=10,
            step=1,
            label="Number of models",
            interactive=True,
        )
        # TODO: add more filters
    get_model_btn = gr.Button(value="Get Models")

    hf_models = gr.Dropdown(
        label="Hugging Face Model List",
        choices=None,
        value=None,
        visible=False,
    )
    # TODO: select and SendTo
    civit_models = gr.Gallery(
        label="Civitai Model Gallery",
        value=None,
        interactive=True,
        visible=False,
    )

    with gr.Row(visible=False) as sendto_btns:
        modelmanager_sendto_txt2img = gr.Button(value="SendTo Txt2Img")
        modelmanager_sendto_img2img = gr.Button(value="SendTo Img2Img")
        modelmanager_sendto_inpaint = gr.Button(value="SendTo Inpaint")
        modelmanager_sendto_outpaint = gr.Button(value="SendTo Outpaint")
        modelmanager_sendto_upscaler = gr.Button(value="SendTo Upscaler")

    def get_model_list(model_source, model_number):
        if model_source == "Hugging Face":
            hf_model_list = get_hf_list(model_number)
            models = []
            for model in hf_model_list:
                # TODO: add model info
                models.append(f'{model["modelId"]}')
            return (
                gr.Dropdown.update(choices=models, visible=True),
                gr.Gallery.update(value=None, visible=False),
                gr.Row.update(visible=True),
            )
        elif model_source == "Civitai":
            civit_model_list = get_civit_list(model_number)
            models = []
            for model in civit_model_list:
                image = get_image_from_model(model)
                if image is None:
                    continue
                # TODO: add model info
                models.append(
                    (Image.open(image), f'{model["files"][0]["downloadUrl"]}')
                )
            return (
                gr.Dropdown.update(value=None, choices=None, visible=False),
                gr.Gallery.update(value=models, visible=True),
                gr.Row.update(visible=False),
            )
        else:
            return (
                gr.Dropdown.update(value=None, choices=None, visible=False),
                gr.Gallery.update(value=None, visible=False),
                gr.Row.update(visible=False),
            )

    get_model_btn.click(
        fn=get_model_list,
        inputs=[model_source, model_number],
        outputs=[
            hf_models,
            civit_models,
            sendto_btns,
        ],
    )
