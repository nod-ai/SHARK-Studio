import os
import sys

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"

import gradio as gr
from apps.stable_diffusion.src import args, clear_all
from apps.stable_diffusion.web.utils.gradio_configs import (
    clear_gradio_tmp_imgs_folder,
)

# clear all gradio tmp images from the last session
clear_gradio_tmp_imgs_folder()

if args.clear_all:
    clear_all()


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_path, relative_path)


dark_theme = resource_path("ui/css/sd_dark_theme.css")

from apps.stable_diffusion.web.ui import (
    txt2img_web,
    txt2img_output,
    txt2img_sendto_img2img,
    txt2img_sendto_inpaint,
    txt2img_sendto_outpaint,
    img2img_web,
    img2img_output,
    img2img_init_image,
    img2img_sendto_inpaint,
    img2img_sendto_outpaint,
    inpaint_web,
    inpaint_output,
    inpaint_init_image,
    inpaint_sendto_img2img,
    inpaint_sendto_outpaint,
    outpaint_web,
    outpaint_output,
    outpaint_init_image,
    outpaint_sendto_img2img,
    outpaint_sendto_inpaint,
)


def register_button_click(button, selectedid, inputs, outputs):
    button.click(
        lambda x: (
            x,
            gr.Tabs.update(selected=selectedid),
        ),
        inputs,
        outputs,
    )


with gr.Blocks(
    css=dark_theme, analytics_enabled=False, title="Stable Diffusion"
) as sd_web:
    with gr.Tabs() as tabs:
        with gr.TabItem(label="Text-to-Image", id=0):
            txt2img_web.render()
        with gr.TabItem(label="Image-to-Image", id=1):
            img2img_web.render()
        with gr.TabItem(label="Inpainting", id=2):
            inpaint_web.render()
        with gr.TabItem(label="Outpainting", id=3):
            outpaint_web.render()

    register_button_click(
        txt2img_sendto_img2img,
        1,
        [txt2img_output],
        [img2img_init_image, tabs],
    )
    register_button_click(
        txt2img_sendto_inpaint,
        2,
        [txt2img_output],
        [inpaint_init_image, tabs],
    )
    register_button_click(
        txt2img_sendto_outpaint,
        3,
        [txt2img_output],
        [outpaint_init_image, tabs],
    )
    register_button_click(
        img2img_sendto_inpaint,
        2,
        [txt2img_output],
        [inpaint_init_image, tabs],
    )
    register_button_click(
        img2img_sendto_outpaint,
        3,
        [txt2img_output],
        [outpaint_init_image, tabs],
    )


sd_web.queue()
sd_web.launch(
    share=args.share,
    inbrowser=True,
    server_name="0.0.0.0",
    server_port=args.server_port,
)
