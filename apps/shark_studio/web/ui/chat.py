import gradio as gr
import time
import os
from pathlib import Path
from datetime import datetime as dt
import json
import sys
from apps.shark_studio.api.utils import (
    get_available_devices,
)
from apps.shark_studio.api.llm import (
    llm_model_map,
    LanguageModel,
)


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


language_model = None


def create_prompt(model_name, history, prompt_prefix):
    return ""
    system_message = ""
    if "llama2" in model_name:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        conversation = "".join(
            [f"{B_INST} {item[0]} {E_INST} {item[1]} " for item in history[1:]]
        )
        if prompt_prefix:
            msg = f"{B_INST} {B_SYS}{system_message}{E_SYS}{history[0][0]} {E_INST} {history[0][1]} {conversation}"
        else:
            msg = f"{B_INST} {history[0][0]} {E_INST} {history[0][1]} {conversation}"
    elif model_name in ["vicuna"]:
        conversation = "".join(
            [
                "".join(["<|USER|>" + item[0], "<|ASSISTANT|>" + item[1]])
                for item in history
            ]
        )
        msg = system_message + conversation
        msg = msg.strip()
    else:
        conversation = "".join(
            ["".join([item[0], item[1]]) for item in history]
        )
        msg = system_message + conversation
        msg = msg.strip()
    return msg


def get_default_config():
    return False
    import torch
    from transformers import AutoTokenizer

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
    from apps.language_models.src.model_wrappers.vicuna_model import (
        CombinedModel,
    )
    from shark.shark_generate_model_config import GenerateConfigFile

    model = CombinedModel()
    c = GenerateConfigFile(model, 1, ["gpu_id"], firstVicunaCompileInput)
    c.split_into_layers()


# model_vmfb_key = ""


def chat_fn(
    prompt_prefix,
    history,
    model,
    device,
    precision,
    download_vmfb,
    config_file,
    cli=False,
):
    global language_model
    if language_model is None:
        history[-1][-1] = "Getting the model ready..."
        yield history, ""
        language_model = LanguageModel(
            model,
            device=device,
            precision=precision,
            external_weights="safetensors",
            use_system_prompt=prompt_prefix,
        )
        history[-1][-1] = "Getting the model ready... Done"
        yield history, ""
        history[-1][-1] = ""
    token_count = 0
    total_time = 0.001  # In order to avoid divide by zero error
    prefill_time = 0
    is_first = True
    for text, exec_time in language_model.chat(history):
        history[-1][-1] = text
        if is_first:
            prefill_time = exec_time
            is_first = False
            yield history, f"Prefill: {prefill_time:.2f}"
        else:
            total_time += exec_time
            token_count += 1
            tokens_per_sec = token_count / total_time
            yield history, f"Prefill: {prefill_time:.2f} seconds\n Decode: {tokens_per_sec:.2f} tokens/sec"


def llm_chat_api(InputData: dict):
    return None
    print(f"Input keys : {InputData.keys()}")
    # print(f"model : {InputData['model']}")
    is_chat_completion_api = (
        "messages" in InputData.keys()
    )  # else it is the legacy `completion` api
    # For Debugging input data from API
    # if is_chat_completion_api:
    #     print(f"message -> role : {InputData['messages'][0]['role']}")
    #     print(f"message -> content : {InputData['messages'][0]['content']}")
    # else:
    #     print(f"prompt : {InputData['prompt']}")
    # print(f"max_tokens : {InputData['max_tokens']}") # Default to 128 for now
    global vicuna_model
    model_name = (
        InputData["model"] if "model" in InputData.keys() else "codegen"
    )
    model_path = llm_model_map[model_name]
    device = "cpu-task"
    precision = "fp16"
    max_toks = (
        None
        if "max_tokens" not in InputData.keys()
        else InputData["max_tokens"]
    )
    if max_toks is None:
        max_toks = 128 if model_name == "codegen" else 512

    # make it working for codegen first
    from apps.language_models.scripts.vicuna import (
        UnshardedVicuna,
    )

    device_id = None
    if vicuna_model == 0:
        if "cuda" in device:
            device = "cuda"
        elif "sync" in device:
            device = "cpu-sync"
        elif "task" in device:
            device = "cpu-task"
        elif "vulkan" in device:
            device_id = int(device.split("://")[1])
            device = "vulkan"
        else:
            print("unrecognized device")

        vicuna_model = UnshardedVicuna(
            model_name,
            hf_model_path=model_path,
            device=device,
            precision=precision,
            max_num_tokens=max_toks,
            download_vmfb=True,
            load_mlir_from_shark_tank=True,
            device_id=device_id,
        )

    # TODO: add role dict for different models
    if is_chat_completion_api:
        # TODO: add funtionality for multiple messages
        prompt = create_prompt(
            model_name, [(InputData["messages"][0]["content"], "")]
        )
    else:
        prompt = InputData["prompt"]
    print("prompt = ", prompt)

    res = vicuna_model.generate(prompt)
    res_op = None
    for op in res:
        res_op = op

    if is_chat_completion_api:
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": res_op,  # since we are yeilding the result
                },
                "finish_reason": "stop",  # or length
            }
        ]
    else:
        choices = [
            {
                "text": res_op,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",  # or length
            }
        ]
    end_time = dt.now().strftime("%Y%m%d%H%M%S%f")
    return {
        "id": end_time,
        "object": "chat.completion"
        if is_chat_completion_api
        else "text_completion",
        "created": int(end_time),
        "choices": choices,
    }


def view_json_file(file_obj):
    content = ""
    with open(file_obj.name, "r") as fopen:
        content = fopen.read()
    return content


with gr.Blocks(title="Chat") as chat_element:
    with gr.Row():
        model_choices = list(llm_model_map.keys())
        model = gr.Dropdown(
            label="Select Model",
            value=model_choices[0],
            choices=model_choices,
            allow_custom_value=True,
        )
        supported_devices = get_available_devices()
        enabled = True
        if len(supported_devices) == 0:
            supported_devices = ["cpu-task"]
        supported_devices = [x for x in supported_devices if "sync" not in x]
        device = gr.Dropdown(
            label="Device",
            value=supported_devices[0],
            choices=supported_devices,
            interactive=enabled,
            allow_custom_value=True,
        )
        precision = gr.Radio(
            label="Precision",
            value="int4",
            choices=[
                # "int4",
                # "int8",
                # "fp16",
                "fp32",
            ],
            visible=False,
        )
        tokens_time = gr.Textbox(label="Tokens generated per second")
        with gr.Column():
            download_vmfb = gr.Checkbox(
                label="Download vmfb from Shark tank if available",
                value=True,
                interactive=True,
            )
            prompt_prefix = gr.Checkbox(
                label="Add System Prompt",
                value=False,
                interactive=True,
            )

    chatbot = gr.Chatbot(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
                interactive=enabled,
                container=False,
            )
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit", interactive=enabled)
                stop = gr.Button("Stop", interactive=enabled)
                clear = gr.Button("Clear", interactive=enabled)

    with gr.Row(visible=False):
        with gr.Group():
            config_file = gr.File(
                label="Upload sharding configuration", visible=False
            )
            json_view_button = gr.Button("View as JSON", visible=False)
        json_view = gr.JSON(visible=False)
        json_view_button.click(
            fn=view_json_file, inputs=[config_file], outputs=[json_view]
        )
    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        show_progress=False,
        queue=False,
    ).then(
        fn=chat_fn,
        inputs=[
            prompt_prefix,
            chatbot,
            model,
            device,
            precision,
            download_vmfb,
            config_file,
        ],
        outputs=[chatbot, tokens_time],
        show_progress=False,
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        show_progress=False,
        queue=False,
    ).then(
        fn=chat_fn,
        inputs=[
            prompt_prefix,
            chatbot,
            model,
            device,
            precision,
            download_vmfb,
            config_file,
        ],
        outputs=[chatbot, tokens_time],
        show_progress=False,
        queue=True,
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, [chatbot], queue=False)
