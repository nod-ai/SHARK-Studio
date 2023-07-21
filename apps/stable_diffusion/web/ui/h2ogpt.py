import gradio as gr
import torch
import os
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
)
from apps.stable_diffusion.web.ui.utils import available_devices

from apps.language_models.langchain.enums import (
    DocumentChoices,
    LangChainAction,
)
import apps.language_models.langchain.gen as gen
from apps.stable_diffusion.src import args


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


sharkModel = 0
sharded_model = 0
h2ogpt_model = 0

past_key_values = None

# NOTE: Each `model_name` should have its own start message
start_message = """
    SHARK DocuChat
    Chat with an AI, contextualized with provided files.
"""


def create_prompt(history):
    system_message = start_message

    conversation = "".join(["".join([item[0], item[1]]) for item in history])

    msg = system_message + conversation
    msg = msg.strip()
    return msg


def chat(curr_system_message, history, model, device, precision):
    args.run_docuchat_web = True
    global sharded_model
    global past_key_values
    global h2ogpt_model
    global userpath_selector

    model_name, model_path = list(map(str.strip, model.split("=>")))
    print(f"In chat for {model_name}")

    # if h2ogpt_model == 0:
    #     if "cuda" in device:
    #         device = "cuda"
    #     elif "sync" in device:
    #         device = "cpu-sync"
    #     elif "task" in device:
    #         device = "cpu-task"
    #     elif "vulkan" in device:
    #         device = "vulkan"
    #     else:
    #         print("unrecognized device")

    #     max_toks = 128 if model_name == "codegen" else 512
    #     h2ogpt_model = UnshardedVicuna(
    #         model_name,
    #         hf_model_path=model_path,
    #         device=device,
    #         precision=precision,
    #         max_num_tokens=max_toks,
    #     )
    prompt = create_prompt(history)
    print(prompt)
    # print("prompt = ", prompt)

    # for partial_text in h2ogpt_model.generate(prompt):
    #     history[-1][1] = partial_text
    #     yield history
    model, tokenizer, device = gen.get_model(
        load_half=True,
        load_gptq="",
        use_safetensors=False,
        infer_devices=True,
        base_model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
        inference_server="",
        tokenizer_base_model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
        lora_weights="",
        gpu_id=0,
        reward_type=None,
        local_files_only=False,
        resume_download=True,
        use_auth_token=False,
        trust_remote_code=True,
        offload_folder=None,
        compile_model=False,
        verbose=False,
    )
    print(tokenizer.model_max_length)
    model_state = dict(
        model=model,
        tokenizer=tokenizer,
        device=device,
        base_model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
        tokenizer_base_model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
        lora_weights="",
        inference_server="",
        prompt_type=None,
        prompt_dict=None,
    )
    output = gen.evaluate(
        model_state,  # model_state
        None,  # my_db_state
        prompt,  # instruction
        "",  # iinput
        "",  # context
        True,  # stream_output
        "prompt_answer",  # prompt_type
        {
            "promptA": "",
            "promptB": "",
            "PreInstruct": "<|prompt|>",
            "PreInput": None,
            "PreResponse": "<|answer|>",
            "terminate_response": [
                "<|prompt|>",
                "<|answer|>",
                "<|endoftext|>",
            ],
            "chat_sep": "<|endoftext|>",
            "chat_turn_sep": "<|endoftext|>",
            "humanstr": "<|prompt|>",
            "botstr": "<|answer|>",
            "generates_leading_space": False,
        },  # prompt_dict
        0.1,  # temperature
        0.75,  # top_p
        40,  # top_k
        1,  # num_beams
        256,  # max_new_tokens
        0,  # min_new_tokens
        False,  # early_stopping
        180,  # max_time
        1.07,  # repetition_penalty
        1,  # num_return_sequences
        False,  # do_sample
        True,  # chat
        prompt,  # instruction_nochat
        "",  # iinput_nochat
        "UserData",  # langchain_mode
        LangChainAction.QUERY.value,  # langchain_action
        3,  # top_k_docs
        True,  # chunk
        512,  # chunk_size
        [DocumentChoices.All_Relevant.name],  # document_choice
        concurrency_count=1,
        memory_restriction_level=2,
        raise_generate_gpu_exceptions=False,
        chat_context="",
        use_openai_embedding=False,
        use_openai_model=False,
        hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        db_type="chroma",
        n_jobs=-1,
        first_para=False,
        max_max_time=60 * 2,
        model_state0=model_state,
        model_lock=True,
        user_path=userpath_selector.value,
    )
    for partial_text in output:
        history[-1][1] = partial_text
        yield history

    return history


with gr.Blocks(title="H2OGPT") as h2ogpt_web:
    with gr.Row():
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
        userpath_selector = gr.Textbox(
            label="Document Directory",
            value=str(
                os.path.abspath("apps/language_models/langchain/user_path/")
            ),
            interactive=True,
            container=True,
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
        inputs=[system_msg, chatbot, device, precision],
        outputs=[chatbot],
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat,
        inputs=[system_msg, chatbot, device, precision],
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
