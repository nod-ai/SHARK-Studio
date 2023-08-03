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
h2ogpt_model = 0


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


def chat(curr_system_message, history, device, precision):
    args.run_docuchat_web = True
    global h2ogpt_model
    global h2ogpt_tokenizer
    global model_state
    global langchain
    global userpath_selector

    if h2ogpt_model == 0:
        if "cuda" in device:
            shark_device = "cuda"
        elif "sync" in device:
            shark_device = "cpu"
        elif "task" in device:
            shark_device = "cpu"
        elif "vulkan" in device:
            shark_device = "vulkan"
        else:
            print("unrecognized device")

        device = "cpu" if shark_device == "cpu" else "cuda"

        args.device = shark_device
        args.precision = precision

        from apps.language_models.langchain.gen import Langchain

        langchain = Langchain(device, precision)
        h2ogpt_model, h2ogpt_tokenizer, _ = langchain.get_model(
            load_4bit=True
            if device == "cuda"
            else False,  # load model in 4bit if device is cuda to save memory
            load_gptq="",
            use_safetensors=False,
            infer_devices=True,
            device=device,
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
        model_state = dict(
            model=h2ogpt_model,
            tokenizer=h2ogpt_tokenizer,
            device=device,
            base_model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
            tokenizer_base_model="h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
            lora_weights="",
            inference_server="",
            prompt_type=None,
            prompt_dict=None,
        )

    prompt = create_prompt(history)
    output = langchain.evaluate(
        model_state=model_state,
        my_db_state=None,
        instruction=prompt,
        iinput="",
        context="",
        stream_output=True,
        prompt_type="prompt_answer",
        prompt_dict={
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
        },
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
        min_new_tokens=0,
        early_stopping=False,
        max_time=180,
        repetition_penalty=1.07,
        num_return_sequences=1,
        do_sample=False,
        chat=True,
        instruction_nochat=prompt,
        iinput_nochat="",
        langchain_mode="UserData",
        langchain_action=LangChainAction.QUERY.value,
        top_k_docs=3,
        chunk=True,
        chunk_size=512,
        document_choice=[DocumentChoices.All_Relevant.name],
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
    history[-1][1] = output["response"]
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
