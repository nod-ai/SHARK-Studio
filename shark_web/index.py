from models.resnet50 import resnet_inf
from models.albert_maskfill import albert_maskfill_inf
import gradio as gr

shark_web = gr.Blocks()

with shark_web:
    image = gr.Image()
    label1 = gr.Label()
    resnet = gr.Button("Recognize Image")

    text = gr.Textbox()
    label2 = gr.Label()
    albert_mask = gr.Button("Decode Mask")

    resnet.click(resnet_inf, inputs=image, outputs=label1)
    albert_mask.click(albert_maskfill_inf, inputs=text, outputs=label2)

shark_web.launch(share=True)
