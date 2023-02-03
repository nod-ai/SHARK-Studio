import gradio as gr
import json
import jsonlines
import os
from args import args
from pathlib import Path
from PIL import Image
from utils import get_datasets


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

    datasets, images, ds_w_prompts = get_datasets(args.gs_url)
    prompt_data = dict()

    with gr.Row(elem_id="ui_body"):
        # TODO: add multiselect dataset, there is a gradio version conflict
        dataset = gr.Dropdown(label="Dataset", choices=datasets)
        image_name = gr.Dropdown(label="Image", choices=[])

    with gr.Row(elem_id="ui_body"):
        # TODO: add ability to search image by typing
        with gr.Column(scale=1, min_width=600):
            image = gr.Image(type="filepath").style(height=512)

        with gr.Column(scale=1, min_width=600):
            prompts = gr.Dropdown(
                label="Prompts",
                choices=[],
            )
            prompt = gr.Textbox(
                label="Editor",
                lines=3,
            )
            with gr.Row():
                save = gr.Button("Save")
                delete = gr.Button("Delete")
            with gr.Row():
                back_image = gr.Button("Back")
                next_image = gr.Button("Next")
            finish = gr.Button("Finish")

    def filter_datasets(dataset):
        if dataset is None:
            return gr.Dropdown.update(value=None, choices=[])

        # create the dataset dir if doesn't exist and download prompt file
        dataset_path = str(shark_root) + "/dataset/" + dataset
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)

        # read prompt jsonlines file
        prompt_data.clear()
        if dataset in ds_w_prompts:
            prompt_gs_path = args.gs_url + "/" + dataset + "/metadata.jsonl"
            os.system(f'gsutil cp "{prompt_gs_path}" "{dataset_path}"/')
            with jsonlines.open(dataset_path + "/metadata.jsonl") as reader:
                for line in reader.iter(type=dict, skip_invalid=True):
                    prompt_data[line["file_name"]] = (
                        [line["text"]]
                        if type(line["text"]) is str
                        else line["text"]
                    )

        return gr.Dropdown.update(choices=images[dataset])

    dataset.change(fn=filter_datasets, inputs=dataset, outputs=image_name)

    def display_image(dataset, image_name):
        if dataset is None or image_name is None:
            return gr.Image.update(value=None), gr.Dropdown.update(value=None)

        # download and load the image
        img_gs_path = args.gs_url + "/" + dataset + "/" + image_name
        img_sub_path = "/".join(image_name.split("/")[:-1])
        img_dst_path = (
            str(shark_root) + "/dataset/" + dataset + "/" + img_sub_path + "/"
        )
        if not os.path.exists(img_dst_path):
            os.mkdir(img_dst_path)
        os.system(f'gsutil cp "{img_gs_path}" "{img_dst_path}"')
        img = Image.open(img_dst_path + image_name.split("/")[-1])

        if image_name not in prompt_data.keys():
            prompt_data[image_name] = []
        prompt_choices = ["Add new"]
        prompt_choices += prompt_data[image_name]
        return gr.Image.update(value=img), gr.Dropdown.update(
            choices=prompt_choices
        )

    image_name.change(
        fn=display_image,
        inputs=[dataset, image_name],
        outputs=[image, prompts],
    )

    def edit_prompt(prompts):
        if prompts == "Add new":
            return gr.Textbox.update(value=None)

        return gr.Textbox.update(value=prompts)

    prompts.change(fn=edit_prompt, inputs=prompts, outputs=prompt)

    def save_prompt(dataset, image_name, prompts, prompt):
        if (
            dataset is None
            or image_name is None
            or prompts is None
            or prompt is None
        ):
            return

        if prompts == "Add new":
            prompt_data[image_name].append(prompt)
        else:
            idx = prompt_data[image_name].index(prompts)
            prompt_data[image_name][idx] = prompt

        prompt_path = (
            str(shark_root) + "/dataset/" + dataset + "/metadata.jsonl"
        )
        # write prompt jsonlines file
        with open(prompt_path, "w") as f:
            for key, value in prompt_data.items():
                if not value:
                    continue
                v = value if len(value) > 1 else value[0]
                f.write(json.dumps({"file_name": key, "text": v}))
                f.write("\n")

        prompt_choices = ["Add new"]
        prompt_choices += prompt_data[image_name]
        return gr.Dropdown.update(choices=prompt_choices, value=None)

    save.click(
        fn=save_prompt,
        inputs=[dataset, image_name, prompts, prompt],
        outputs=prompts,
    )

    def delete_prompt(dataset, image_name, prompts):
        if dataset is None or image_name is None or prompts is None:
            return
        if prompts == "Add new":
            return

        prompt_data[image_name].remove(prompts)
        prompt_path = (
            str(shark_root) + "/dataset/" + dataset + "/metadata.jsonl"
        )
        # write prompt jsonlines file
        with open(prompt_path, "w") as f:
            for key, value in prompt_data.items():
                if not value:
                    continue
                v = value if len(value) > 1 else value[0]
                f.write(json.dumps({"file_name": key, "text": v}))
                f.write("\n")

        prompt_choices = ["Add new"]
        prompt_choices += prompt_data[image_name]
        return gr.Dropdown.update(choices=prompt_choices, value=None)

    delete.click(
        fn=delete_prompt,
        inputs=[dataset, image_name, prompts],
        outputs=prompts,
    )

    def get_back_image(dataset, image_name):
        if dataset is None or image_name is None:
            return

        # remove local image
        img_path = str(shark_root) + "/dataset/" + dataset + "/" + image_name
        os.system(f'rm "{img_path}"')
        # get the index for the back image
        idx = images[dataset].index(image_name)
        if idx == 0:
            return gr.Dropdown.update(value=None)

        return gr.Dropdown.update(value=images[dataset][idx - 1])

    back_image.click(
        fn=get_back_image, inputs=[dataset, image_name], outputs=image_name
    )

    def get_next_image(dataset, image_name):
        if dataset is None or image_name is None:
            return

        # remove local image
        img_path = str(shark_root) + "/dataset/" + dataset + "/" + image_name
        os.system(f'rm "{img_path}"')
        # get the index for the next image
        idx = images[dataset].index(image_name)
        if idx == len(images[dataset]) - 1:
            return gr.Dropdown.update(value=None)

        return gr.Dropdown.update(value=images[dataset][idx + 1])

    next_image.click(
        fn=get_next_image, inputs=[dataset, image_name], outputs=image_name
    )

    def finish_annotation(dataset):
        if dataset is None:
            return

        # upload prompt and remove local data
        dataset_path = str(shark_root) + "/dataset/" + dataset
        dataset_gs_path = args.gs_url + "/" + dataset + "/"
        os.system(
            f'gsutil cp "{dataset_path}/metadata.jsonl" "{dataset_gs_path}"'
        )
        os.system(f'rm -rf "{dataset_path}"')

        return gr.Dropdown.update(value=None)

    finish.click(fn=finish_annotation, inputs=dataset, outputs=dataset)


if __name__ == "__main__":
    shark_web.launch(
        share=args.share,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=args.server_port,
    )
