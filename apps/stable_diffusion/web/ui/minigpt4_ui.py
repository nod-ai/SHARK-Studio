# ========================================
#             Gradio Setting
# ========================================
import gradio as gr
from apps.language_models.src.pipelines.minigpt4_pipeline import (
    MiniGPT4,
    CONV_VISION,
)

chat = None


def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return (
        None,
        gr.update(value=None, interactive=True),
        gr.update(
            placeholder="Please upload your image first", interactive=False
        ),
        gr.update(value="Upload & Start Chat", interactive=True),
        chat_state,
        img_list,
    )


def upload_img(gr_img, text_input, chat_state, device):
    global chat
    if chat is None:
        from apps.language_models.src.pipelines.minigpt4_pipeline import (
            MiniGPT4,
        )

        chat = MiniGPT4(
            model_name="MiniGPT4",
            hf_model_path=None,
            max_new_tokens=30,
            device=device,
            precision="fp16",
        )
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return (
        gr.update(interactive=False),
        gr.update(interactive=True, placeholder="Type and press Enter"),
        gr.update(value="Start Chatting", interactive=False),
        chat_state,
        img_list,
    )


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return (
            gr.update(
                interactive=True, placeholder="Input should not be empty!"
            ),
            chatbot,
            chat_state,
        )
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return "", chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000,
    )[0]
    print(llm_message)
    print("************")
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list


title = """<h1 align="center">MultiModal SHARK (experimental)</h1>"""
description = """<h3>Upload your images and start chatting!</h3>"""
article = """<p><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
"""

# TODO show examples below

with gr.Blocks() as minigpt4_web:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(
                value="Upload & Start Chat",
                interactive=True,
                variant="primary",
            )
            clear = gr.Button("Restart")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

            device = gr.Dropdown(
                label="Device",
                value="cuda",
                # if enabled
                # else "Only CUDA Supported for now",
                choices=["cuda"],
                interactive=False,
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label="MiniGPT-4")
            text_input = gr.Textbox(
                label="User",
                placeholder="Please upload your image first",
                interactive=False,
            )

    upload_button.click(
        upload_img,
        [image, text_input, chat_state, device],
        [image, text_input, upload_button, chat_state, img_list],
    )

    text_input.submit(
        gradio_ask,
        [text_input, chatbot, chat_state],
        [text_input, chatbot, chat_state],
    ).then(
        gradio_answer,
        [chatbot, chat_state, img_list, num_beams, temperature],
        [chatbot, chat_state, img_list],
    )
    clear.click(
        gradio_reset,
        [chat_state, img_list],
        [chatbot, image, text_input, upload_button, chat_state, img_list],
        queue=False,
    )
