import gradio as gr
import torch
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
)
from apps.stable_diffusion.web.ui.utils import available_devices
from datetime import datetime as dt


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


sharkModel = 0
sharded_model = 0
vicuna_model = 0

past_key_values = None

model_map = {
    "llama2_7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2_70b": "meta-llama/Llama-2-70b-chat-hf",
    "codegen": "Salesforce/codegen25-7b-multi",
    "vicuna1p3": "lmsys/vicuna-7b-v1.3",
    "vicuna": "TheBloke/vicuna-7B-1.1-HF",
    "StableLM": "stabilityai/stablelm-tuned-alpha-3b",
}

# NOTE: Each `model_name` should have its own start message
start_message = {
    "llama2": (
        "System: You are a helpful, respectful and honest assistant. Always answer "
        "as helpfully as possible, while being safe.  Your answers should not "
        "include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
        "content. Please ensure that your responses are socially unbiased and positive "
        "in nature. If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. If you don't know the "
        "answer to a question, please don't share false information."
    ),
    "StableLM": (
        "<|SYSTEM|># StableLM Tuned (Alpha version)"
        "\n- StableLM is a helpful and harmless open-source AI language model "
        "developed by StabilityAI."
        "\n- StableLM is excited to be able to help the user, but will refuse "
        "to do anything that could be considered harmful to the user."
        "\n- StableLM is more than just an information source, StableLM is also "
        "able to write poetry, short stories, and make jokes."
        "\n- StableLM will refuse to participate in anything that "
        "could harm a human."
    ),
    "vicuna": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions.\n"
    ),
    "vicuna1p3": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions.\n"
    ),
    "codegen": "",
}


def create_prompt(model_name, history):
    system_message = start_message[model_name]

    if model_name in ["StableLM", "vicuna", "vicuna1p3", "llama2"]:
        conversation = "".join(
            [
                "".join(["<|USER|>" + item[0], "<|ASSISTANT|>" + item[1]])
                for item in history
            ]
        )
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


# TODO: Make chat reusable for UI and API
def chat(curr_system_message, history, model, device, precision, cli=True):
    global past_key_values

    global vicuna_model
    model_name, model_path = list(map(str.strip, model.split("=>")))

    if model_name in ["vicuna", "vicuna1p3", "codegen", "llama2"]:
        from apps.language_models.scripts.vicuna import (
            UnshardedVicuna,
        )

        if vicuna_model == 0:
            if "cuda" in device:
                device = "cuda"
            elif "sync" in device:
                device = "cpu-sync"
            elif "task" in device:
                device = "cpu-task"
            elif "vulkan" in device:
                device = "vulkan"
            else:
                print("unrecognized device")

            max_toks = 128 if model_name == "codegen" else 512
            vicuna_model = UnshardedVicuna(
                model_name,
                hf_model_path=model_path,
                device=device,
                precision=precision,
                max_num_tokens=max_toks,
            )
        prompt = create_prompt(model_name, history)

        for partial_text in vicuna_model.generate(prompt, cli=cli):
            history[-1][1] = partial_text
            yield history

        return history

    # else Model is StableLM
    global sharkModel
    from apps.language_models.src.pipelines.stablelm_pipeline import (
        SharkStableLM,
    )

    if sharkModel == 0:
        # max_new_tokens=512
        shark_slm = SharkStableLM(
            model_name
        )  # pass elements from UI as required

    # Construct the input message string for the model by concatenating the
    # current system message and conversation history
    if len(curr_system_message.split()) > 160:
        print("clearing context")
    prompt = create_prompt(model_name, history)
    generate_kwargs = dict(prompt=prompt)

    words_list = shark_slm.generate(**generate_kwargs)

    partial_text = ""
    for new_text in words_list:
        print(new_text)
        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to clean up the message textbox and the updated
        # conversation history
        yield history
    return words_list


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

    if vicuna_model == 0:
        if "cuda" in device:
            device = "cuda"
        elif "sync" in device:
            device = "cpu-sync"
        elif "task" in device:
            device = "cpu-task"
        elif "vulkan" in device:
            device = "vulkan"
        else:
            print("unrecognized device")

        vicuna_model = UnshardedVicuna(
            model_name,
            hf_model_path=model_path,
            device=device,
            precision=precision,
            max_num_tokens=max_toks,
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
        print(supported_devices)
        device = gr.Dropdown(
            label="Device",
            value=supported_devices[0]
            if enabled
            else "Only CUDA Supported for now",
            choices=supported_devices,
            interactive=enabled,
        )
        precision = gr.Radio(
            label="Precision",
            value="fp16",
            choices=[
                "int4",
                "int8",
                "fp16",
                "fp32",
            ],
            visible=True,
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
        inputs=[system_msg, chatbot, model, device, precision],
        outputs=[chatbot],
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat,
        inputs=[system_msg, chatbot, model, device, precision],
        outputs=[chatbot],
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
