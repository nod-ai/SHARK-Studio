import gradio as gr
import json
import jsonlines
import os
from pathlib import Path
from PIL import Image
from utils import get_datasets


# see https://cloud.google.com/docs/authentication/provide-credentials-adc to authorize
gs_url = "gs://shark-datasets/portraits"

shark_root = Path(__file__).parent.parent
demo_css = shark_root.joinpath("web/demo.css").resolve()
nodlogo_loc = shark_root.joinpath(
    "web/models/stable_diffusion/logos/nod-logo.png"
)


with gr.Blocks(title="Dataset Annotation Tool", css=demo_css) as shark_web:

    with gr.Row(elem_id="ui_title"):
        nod_logo = Image.open(nodlogo_loc)
        with gr.Column(scale=1, elem_id="demo_title_outer"):
            gr.Image(
                value=nod_logo,
                show_label=False,
                interactive=False,
                elem_id="top_logo",
            ).style(width=150, height=100)

    datasets, images = get_datasets(gs_url)
    prompt_data = dict()

    with gr.Row(elem_id="ui_body"):
        # TODO: add multiselect dataset
        dataset = gr.Dropdown(label="Dataset", choices=datasets)
        image_name = gr.Dropdown(label="Image", choices=[])

    with gr.Row(elem_id="ui_body", visible=True):
        # TODO: add ability to search image by typing
        with gr.Column(scale=1, min_width=600):
            image = gr.Image(type="filepath").style(height=512)

        with gr.Column(scale=1, min_width=600):
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
            )
            next_image = gr.Button("Next")
            finish = gr.Button("Finish")

    def filter_datasets(dataset):
        # TODO: execute finish process when switching dataset
        if dataset is None:
            return gr.Dropdown.update(value=None, choices=[])

        # create the dataset dir if doesn't exist and download prompt file
        dataset_path = str(shark_root) + "/dataset/" + dataset
        prompt_gs_path = gs_url + "/" + dataset + "/metadata.jsonl"
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        os.system(f'gsutil cp "{prompt_gs_path}" "{dataset_path}"/')

        # read prompt jsonlines file
        prompt_data.clear()
        with jsonlines.open(dataset_path + "/metadata.jsonl") as reader:
            for line in reader.iter(type=dict, skip_invalid=True):
                prompt_data[line["file_name"]] = line["text"]

        return gr.Dropdown.update(choices=images[dataset])

    dataset.change(fn=filter_datasets, inputs=dataset, outputs=image_name)

    def display_image(dataset, image_name):
        if dataset is None or image_name is None:
            return gr.Image.update(value=None), gr.Textbox.update(value=None)

        # download and load the image
        # TODO: remove previous image if change image from dropdown
        img_gs_path = gs_url + "/" + dataset + "/" + image_name
        img_sub_path = "/".join(image_name.split("/")[:-1])
        img_dst_path = (
            str(shark_root) + "/dataset/" + dataset + "/" + img_sub_path + "/"
        )
        if not os.path.exists(img_dst_path):
            os.mkdir(img_dst_path)
        os.system(f'gsutil cp "{img_gs_path}" "{img_dst_path}"')
        img = Image.open(img_dst_path + image_name.split("/")[-1])

        return gr.Image.update(value=img), gr.Textbox.update(
            value=prompt_data[image_name]
        )

    image_name.change(
        fn=display_image, inputs=[dataset, image_name], outputs=[image, prompt]
    )

    def update_prompt(dataset, image_name, prompt):
        if dataset is None or image_name is None or prompt is None:
            return

        prompt_data[image_name] = prompt
        prompt_path = (
            str(shark_root) + "/dataset/" + dataset + "/metadata.jsonl"
        )
        # write prompt jsonlines file
        with open(prompt_path, "w") as f:
            for key, value in prompt_data.items():
                f.write(json.dumps({"file_name": key, "text": value}))
                f.write("\n")
        return

    prompt.change(fn=update_prompt, inputs=[dataset, image_name, prompt])

    def get_next_image(dataset, image_name):
        if dataset is None or image_name is None:
            return

        # remove local image
        img_path = str(shark_root) + "/dataset/" + dataset + "/" + image_name
        os.system(f'rm "{img_path}"')
        # get the index for the next image
        # TODO: finish when get to the end
        idx = images[dataset].index(image_name)

        return gr.Dropdown.update(value=images[dataset][idx + 1])

    next_image.click(
        fn=get_next_image, inputs=[dataset, image_name], outputs=image_name
    )

    def finish_annotation(dataset):
        if dataset is None:
            return

        # upload prompt and remove local data
        dataset_path = str(shark_root) + "/dataset/" + dataset
        dataset_gs_path = gs_url + "/" + dataset + "/"
        os.system(
            f'gsutil cp "{dataset_path}/metadata.jsonl" "{dataset_gs_path}"'
        )
        os.system(f'rm -rf "{dataset_path}"')

        return gr.Dropdown.update(value=None)

    finish.click(fn=finish_annotation, inputs=dataset, outputs=dataset)


if __name__ == "__main__":
    shark_web.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=8080,
    )
