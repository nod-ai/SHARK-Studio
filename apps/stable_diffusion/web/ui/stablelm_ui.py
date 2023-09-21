import gradio as gr
import torch
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
)
from apps.stable_diffusion.web.ui.utils import available_devices
from datetime import datetime as dt
import json
import time


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


sharkModel = 0
sharded_model = 0
vicuna_model = 0

past_key_values = None

model_map = {
    "llama2_7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2_13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2_70b": "meta-llama/Llama-2-70b-chat-hf",
    "vicuna": "TheBloke/vicuna-7B-1.1-HF",
}

# NOTE: Each `model_name` should have its own start message
start_message = {
    "llama2_7b": (
        "System: You are a helpful, respectful and honest assistant. Always answer "
        "as helpfully as possible, while being safe.  Your answers should not "
        "include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
        "content. Please ensure that your responses are socially unbiased and positive "
        "in nature. If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. If you don't know the "
        "answer to a question, please don't share false information."
    ),
    "llama2_13b": (
        "System: You are a helpful, respectful and honest assistant. Always answer "
        "as helpfully as possible, while being safe.  Your answers should not "
        "include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
        "content. Please ensure that your responses are socially unbiased and positive "
        "in nature. If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. If you don't know the "
        "answer to a question, please don't share false information."
    ),
    "llama2_70b": (
        "System: You are a helpful, respectful and honest assistant. Always answer "
        "as helpfully as possible, while being safe.  Your answers should not "
        "include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
        "content. Please ensure that your responses are socially unbiased and positive "
        "in nature. If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. If you don't know the "
        "answer to a question, please don't share false information."
    ),
    "vicuna": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions.\n"
    ),
}


def create_prompt(model_name, history):
    system_message = start_message[model_name]

    if "llama2" in model_name:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        conversation = "".join(
            [f"{B_INST} {item[0]} {E_INST} {item[1]} " for item in history[1:]]
        )
        msg = f"{B_INST} {B_SYS} {system_message} {E_SYS} {history[0][0]} {E_INST} {history[0][1]} {conversation}"
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


def set_vicuna_model(model):
    global vicuna_model
    vicuna_model = model


def get_default_config():
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


model_vmfb_key = ""


# TODO: Make chat reusable for UI and API
def chat(
    curr_system_message,
    history,
    model,
    device,
    precision,
    download_vmfb,
    config_file,
    cli=False,
    progress=gr.Progress(),
):
    global past_key_values
    global model_vmfb_key
    global vicuna_model

    device_id = None
    model_name, model_path = list(map(str.strip, model.split("=>")))
    if "cuda" in device:
        device = "cuda"
    elif "sync" in device:
        device = "cpu-sync"
    elif "task" in device:
        device = "cpu-task"
    elif "vulkan" in device:
        device_id = int(device.split("://")[1])
        device = "vulkan"
    elif "rocm" in device:
        device = "rocm"
    else:
        print("unrecognized device")

    from apps.language_models.scripts.vicuna import ShardedVicuna
    from apps.language_models.scripts.vicuna import UnshardedVicuna
    from apps.stable_diffusion.src import args

    new_model_vmfb_key = f"{model_name}#{model_path}#{device}#{precision}"
    if new_model_vmfb_key != model_vmfb_key:
        model_vmfb_key = new_model_vmfb_key
        max_toks = 128 if model_name == "codegen" else 512

        # get iree flags that need to be overridden, from commandline args
        _extra_args = []
        # vulkan target triple
        vulkan_target_triple = args.iree_vulkan_target_triple
        from shark.iree_utils.vulkan_utils import (
            get_all_vulkan_devices,
            get_vulkan_target_triple,
        )

        if device == "vulkan":
            vulkaninfo_list = get_all_vulkan_devices()
            if vulkan_target_triple == "":
                # We already have the device_id extracted via WebUI, so we directly use
                # that to find the target triple.
                vulkan_target_triple = get_vulkan_target_triple(
                    vulkaninfo_list[device_id]
                )
            _extra_args.append(
                f"-iree-vulkan-target-triple={vulkan_target_triple}"
            )
            if "rdna" in vulkan_target_triple:
                flags_to_add = [
                    "--iree-spirv-index-bits=64",
                ]
                _extra_args = _extra_args + flags_to_add

            if device_id is None:
                id = 0
                for device in vulkaninfo_list:
                    target_triple = get_vulkan_target_triple(
                        vulkaninfo_list[id]
                    )
                    if target_triple == vulkan_target_triple:
                        device_id = id
                        break
                    id += 1

                assert (
                    device_id
                ), f"no vulkan hardware for target-triple '{vulkan_target_triple}' exists"

        print(f"Will use target triple : {vulkan_target_triple}")

        if model_name == "vicuna4":
            vicuna_model = ShardedVicuna(
                model_name,
                hf_model_path=model_path,
                device=device,
                precision=precision,
                max_num_tokens=max_toks,
                compressed=True,
                extra_args_cmd=_extra_args,
            )
        else:
            #  if config_file is None:
            vicuna_model = UnshardedVicuna(
                model_name,
                hf_model_path=model_path,
                hf_auth_token=args.hf_auth_token,
                device=device,
                precision=precision,
                max_num_tokens=max_toks,
                download_vmfb=download_vmfb,
                load_mlir_from_shark_tank=True,
                extra_args_cmd=_extra_args,
                device_id=device_id,
            )

    prompt = create_prompt(model_name, history)

    partial_text = ""
    count = 0
    start_time = time.time()
    for text, msg in progress.tqdm(
        vicuna_model.generate(prompt, cli=cli),
        desc="generating response",
    ):
        count += 1
        if "formatted" in msg:
            history[-1][1] = text
            end_time = time.time()
            tokens_per_sec = count / (end_time - start_time)
            yield history, str(format(tokens_per_sec, ".2f")) + " tokens/sec"
        else:
            partial_text += text + " "
            history[-1][1] = partial_text
            yield history, ""

    return history, ""


def llm_chat_api(InputData: dict):
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
    model_path = model_map[model_name]
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


with gr.Blocks(title="Chatbot") as stablelm_chat:
    with gr.Row():
        model_choices = list(
            map(lambda x: f"{x[0]: <10} => {x[1]}", model_map.items())
        )
        model = gr.Dropdown(
            label="Select Model",
            value=model_choices[0],
            choices=model_choices,
        )
        supported_devices = available_devices
        enabled = len(supported_devices) > 0
        # show cpu-task device first in list for chatbot
        supported_devices = supported_devices[-1:] + supported_devices[:-1]
        supported_devices = [x for x in supported_devices if "sync" not in x]
        device = gr.Dropdown(
            label="Device",
            value=supported_devices[0]
            if enabled
            else "Only CUDA Supported for now",
            choices=supported_devices,
            interactive=enabled,
            # multiselect=True,
        )
        precision = gr.Radio(
            label="Precision",
            value="int8",
            choices=[
                "int4",
                "int8",
                "fp16",
            ],
            visible=True,
        )
        with gr.Column():
            download_vmfb = gr.Checkbox(
                label="Download vmfb from Shark tank if available",
                value=True,
                interactive=True,
            )
            tokens_time = gr.Textbox(label="Tokens generated per second")

    with gr.Row(visible=False):
        with gr.Group():
            config_file = gr.File(
                label="Upload sharding configuration", visible=False
            )
            json_view_button = gr.Button(label="View as JSON", visible=False)
        json_view = gr.JSON(interactive=True, visible=False)
        json_view_button.click(
            fn=view_json_file, inputs=[config_file], outputs=[json_view]
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
    system_msg = gr.Textbox(
        start_message, label="System Message", interactive=False, visible=False
    )

    submit_event = msg.submit(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat,
        inputs=[
            system_msg,
            chatbot,
            model,
            device,
            precision,
            download_vmfb,
            config_file,
        ],
        outputs=[chatbot, tokens_time],
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat,
        inputs=[
            system_msg,
            chatbot,
            model,
            device,
            precision,
            download_vmfb,
            config_file,
        ],
        outputs=[chatbot, tokens_time],
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
