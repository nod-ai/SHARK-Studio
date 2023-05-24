import gradio as gr

from omegaconf import OmegaConf

from apps.stable_diffusion.web.ui.minigpt4.conversation import Chat, CONV_VISION, MiniGPT4SHARK
from apps.stable_diffusion.web.ui.minigpt4.blip_processors import Blip2ImageEvalProcessor

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
config = OmegaConf.load("apps/stable_diffusion/web/ui/minigpt4/configs/minigpt4_eval.yaml")
model_config = OmegaConf.create()
model_config = OmegaConf.merge(
    model_config,
    OmegaConf.load('apps/stable_diffusion/web/ui/minigpt4/configs/minigpt4.yaml'),
    {"model": config["model"]},
)
model_config = model_config['model']
model_config.device_8bit = 0
model = MiniGPT4SHARK.from_config(model_config).to('cpu')

datasets = config.get("datasets", None)
dataset_config = OmegaConf.create()
for dataset_name in datasets:
    dataset_config_path = 'apps/stable_diffusion/web/ui/minigpt4/configs/cc_sbu_align.yaml'
    dataset_config = OmegaConf.merge(
        dataset_config,
        OmegaConf.load(dataset_config_path),
        {"datasets": {dataset_name: config["datasets"][dataset_name]}},
    )
dataset_config = dataset_config['datasets']
vis_processor_cfg = dataset_config.cc_sbu_align.vis_processor.train
vis_processor = Blip2ImageEvalProcessor.from_config(vis_processor_cfg)

llama = model.llama_model

chat = Chat(model, vis_processor, device='cpu')
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list

title = """<h1 align="center">MiniGPT-4 using SHARK</h1>"""
description = """<h3>Upload your images and start chatting!</h3>"""

#TODO show examples below

with gr.Blocks() as minigpt4_web:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
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

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='MiniGPT-4')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)
    
    upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list], queue=False)
