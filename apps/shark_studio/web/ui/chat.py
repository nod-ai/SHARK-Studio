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

B_SYS, E_SYS = "<s>", "</s>"


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def append_bot_prompt(history, input_prompt):
    user_prompt = f"{input_prompt} {E_SYS} {E_SYS}"
    history += user_prompt
    return history


language_model = None


def get_default_config():
    return False


# model_vmfb_key = ""


def chat_fn(
    prompt_prefix,
    history,
    model,
    device,
    precision,
    download_vmfb,
    config_file,
    streaming_llm,
    cli=False,
):
    global language_model
    print("Prompt prefix: ", prompt_prefix)
    if streaming_llm and prompt_prefix == "Clear":
        language_model = None
        return "Clearing history...", ""
    if language_model is None:
        history[-1][-1] = "Getting the model ready..."
        yield history, ""
        language_model = LanguageModel(
            model,
            device=device,
            precision=precision,
            external_weights="safetensors",
            use_system_prompt=prompt_prefix,
            streaming_llm=streaming_llm,
        )
        history[-1][-1] = "Getting the model ready... Done"
        yield history, ""
        history[-1][-1] = ""
    token_count = 0
    total_time = 0.001  # In order to avoid divide by zero error
    prefill_time = 0
    is_first = True
    for text, exec_time in language_model.chat(history):
        history[-1][-1] = f"{text}{E_SYS}"
        if is_first:
            prefill_time = exec_time
            is_first = False
            yield history, f"Prefill: {prefill_time:.2f}"
        else:
            total_time += exec_time
            token_count += 1
            tokens_per_sec = token_count / total_time
            yield history, f"Prefill: {prefill_time:.2f} seconds\n Decode: {tokens_per_sec:.2f} tokens/sec"


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
            value="fp32",
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
                value=False,
                interactive=True,
                visible=False,
            )
            streaming_llm = gr.Checkbox(
                label="Run in streaming mode (requires recompilation)",
                value=True,
                interactive=True,
            )
            prompt_prefix = gr.Checkbox(
                label="Add System Prompt",
                value=True,
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
            config_file = gr.File(label="Upload sharding configuration", visible=False)
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
            streaming_llm,
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
            streaming_llm,
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
    clear.click(
        fn=chat_fn,
        inputs=[
            clear,
            chatbot,
            model,
            device,
            precision,
            download_vmfb,
            config_file,
            streaming_llm,
        ],
        outputs=[chatbot, tokens_time],
        show_progress=False,
        queue=True,
    ).then(lambda: None, None, [chatbot], queue=False)
