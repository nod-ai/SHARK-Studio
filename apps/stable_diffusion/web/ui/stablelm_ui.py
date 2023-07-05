import gradio as gr
import torch
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
)
from apps.stable_diffusion.web.ui.utils import available_devices

start_message = (
    "<|SYSTEM|># StableLM Tuned (Alpha version)"
    "\n- StableLM is a helpful and harmless open-source AI language model "
    "developed by StabilityAI."
    "\n- StableLM is excited to be able to help the user, but will refuse "
    "to do anything that could be considered harmful to the user."
    "\n- StableLM is more than just an information source, StableLM is also "
    "able to write poetry, short stories, and make jokes."
    "\n- StableLM will refuse to participate in anything that "
    "could harm a human."
)


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


sharkModel = 0
sharded_model = 0
vicuna_model = 0


start_message_vicuna = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's "
    "questions.\n"
)
past_key_values = None


def chat(curr_system_message, history, model, device, precision):
    print(f"In chat for {model}")
    global sharded_model
    global past_key_values
    global vicuna_model
    if "vicuna" in model:
        from apps.language_models.src.pipelines.vicuna_pipeline import (
            Vicuna,
        )

        curr_system_message = start_message_vicuna
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
            vicuna_model = Vicuna(
                "vicuna",
                hf_model_path=model,
                device=device,
                precision=precision,
            )
        messages = curr_system_message + "".join(
            [
                "".join(["<|USER|>" + item[0], "<|ASSISTANT|>" + item[1]])
                for item in history
            ]
        )
        prompt = messages.strip()
        print("prompt = ", prompt)

        for partial_text in vicuna_model.generate(prompt):
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
            "StableLM"
        )  # pass elements from UI as required

    # Construct the input message string for the model by concatenating the
    # current system message and conversation history
    if len(curr_system_message.split()) > 160:
        print("clearing context")
        curr_system_message = start_message
    messages = curr_system_message + "".join(
        [
            "".join(["<|USER|>" + item[0], "<|ASSISTANT|>" + item[1]])
            for item in history
        ]
    )

    generate_kwargs = dict(prompt=messages)

    words_list = shark_slm.generate(**generate_kwargs)

    partial_text = ""
    for new_text in words_list:
        # print(new_text)
        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to clean up the message textbox and the updated
        # conversation history
        yield history
    return words_list


with gr.Blocks(title="Chatbot") as stablelm_chat:
    with gr.Row():
        model = gr.Dropdown(
            label="Select Model",
            value="TheBloke/vicuna-7B-1.1-HF",
            choices=[
                "stabilityai/stablelm-tuned-alpha-3b",
                "TheBloke/vicuna-7B-1.1-HF",
            ],
        )
        supported_devices = available_devices
        enabled = len(supported_devices) > 0
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
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
                interactive=enabled,
            ).style(container=False)
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
