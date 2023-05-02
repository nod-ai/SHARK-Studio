import gradio as gr
import torch
import os
from apps.language_models.scripts.stablelm import (
    compile_stableLM,
    StopOnTokens,
    generate,
    sharkModel,
    tok,
    StableLMModel,
)
from transformers import (
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteriaList,
)

start_message = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


input_ids = torch.randint(3, (1, 256))
attention_mask = torch.randint(3, (1, 256))


sharkModel = 0


def chat(curr_system_message, history):
    global sharkModel
    print("In chat")
    if sharkModel == 0:
        # sharkModel = compile_stableLM(None, tuple([input_ids, attention_mask]), "stableLM_linalg_f32_seqLen256", "/home/shark/disk/phaneesh/stablelm_3b_f32_cuda_2048_newflags.vmfb")
        m = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stablelm-tuned-alpha-3b", torch_dtype=torch.float32
        )
        stableLMModel = StableLMModel(m)
        sharkModel = compile_stableLM(
            stableLMModel,
            tuple([input_ids, attention_mask]),
            "stableLM_linalg_f32_seqLen256",
            os.getcwd(),
        )
    # Initialize a StopOnTokens object
    stop = StopOnTokens()
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
    # print(messages)
    # Tokenize the messages string
    streamer = TextIteratorStreamer(
        tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        new_text=messages,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=1.0,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop]),
        sharkStableLM=sharkModel,
    )
    words_list = generate(**generate_kwargs)
    partial_text = ""
    for new_text in words_list:
        # print(new_text)
        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to cleanup the message textbox and the updated conversation history
        yield history
    return words_list


with gr.Blocks(title="StableLM chatbot") as stablelm_chat:
    gr.Markdown("## StableLM-Tuned-Alpha-3b Chat")
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
        fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True
    )
    submit_click_event = submit.click(
        fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False
    ).then(
        fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, [chatbot], queue=False)
