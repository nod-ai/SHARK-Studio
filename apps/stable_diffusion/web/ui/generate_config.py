import gradio as gr
import torch
from transformers import AutoTokenizer
from apps.language_models.src.model_wrappers.vicuna_model import CombinedModel
from shark.shark_generate_model_config import GenerateConfigFile


def get_model_config():
    hf_model_path = "TheBloke/vicuna-7B-1.1-HF"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, use_fast=False)
    compilation_prompt = "".join(["0" for _ in range(17)])
    compilation_input_ids = tokenizer(
        compilation_prompt,
        return_tensors="pt",
    ).input_ids
    compilation_input_ids = torch.tensor(compilation_input_ids).reshape(
        [1, 19]
    )
    firstVicunaCompileInput = (compilation_input_ids,)

    model = CombinedModel()
    c = GenerateConfigFile(model, 1, ["gpu_id"], firstVicunaCompileInput)
    return c.split_into_layers()


with gr.Blocks() as model_config_web:
    with gr.Row():
        hf_models = gr.Dropdown(
            label="Model List",
            choices=["Vicuna"],
            value="Vicuna",
            visible=True,
        )
        get_model_config_btn = gr.Button(value="Get Model Config")
    json_view = gr.JSON()

    get_model_config_btn.click(
        fn=get_model_config,
        inputs=[],
        outputs=[json_view],
    )
