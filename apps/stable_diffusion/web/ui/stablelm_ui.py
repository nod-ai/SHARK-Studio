import gradio as gr
import torch
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
)
from apps.stable_diffusion.web.ui.utils import available_devices

start_message = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


sharkModel = 0
sharded_model = 0


start_message_vicuna = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
past_key_values = None


def chat(curr_system_message, history, model):
    print(f"In chat for {model}")
    global sharded_model
    global past_key_values
    if "vicuna" in model:
        from apps.language_models.scripts.vicuna import (
            tokenizer,
            get_sharded_model,
        )

        SAMPLE_INPUT_LEN = 137
        curr_system_message = start_message_vicuna
        if sharded_model == 0:
            sharded_model = get_sharded_model()
        messages = curr_system_message + "".join(
            [
                "".join(["<|USER|>" + item[0], "<|ASSISTANT|>" + item[1]])
                for item in history
            ]
        )
        prompt = messages.strip()
        print("prompt = ", prompt)
        input_ids = tokenizer(prompt).input_ids
        new_sentence = ""
        for _ in range(200):
            original_input_ids = input_ids
            input_id_len = len(input_ids)
            pad_len = SAMPLE_INPUT_LEN - input_id_len
            attention_mask = torch.ones([1, input_id_len], dtype=torch.int64)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.reshape([1, input_id_len])
            attention_mask = torch.nn.functional.pad(
                torch.tensor(attention_mask),
                (0, pad_len),
                mode="constant",
                value=0,
            )

            if _ == 0:
                output = sharded_model.forward(input_ids, is_first=True)
            else:
                output = sharded_model.forward(
                    input_ids, past_key_values=past_key_values, is_first=False
                )
            logits = output["logits"]
            past_key_values = output["past_key_values"]
            new_word = tokenizer.decode(torch.argmax(logits[:, -1, :], dim=1))
            if new_word == "</s>":
                break
            new_sentence += " " + new_word
            history[-1][1] = new_sentence
            yield history
            next_token = torch.argmax(logits[:, input_id_len - 1, :], dim=1)
            original_input_ids.append(next_token)
            input_ids = [next_token]
        print(new_sentence)
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

    # Construct the input message string for the model by concatenating the current system message and conversation history
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
        # Yield an empty string to cleanup the message textbox and the updated conversation history
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
        device_value = None
        for d in available_devices:
            if "vulkan" in d:
                device_value = d
                break

        device = gr.Dropdown(
            label="Device",
            value=device_value if device_value else available_devices[0],
            interactive=False,
            choices=available_devices,
        )
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    system_msg = gr.Textbox(
        start_message, label="System Message", interactive=False, visible=False
    )

    submit_event = msg.submit(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat,
        inputs=[system_msg, chatbot, model],
        outputs=[chatbot],
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat,
        inputs=[system_msg, chatbot, model],
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
