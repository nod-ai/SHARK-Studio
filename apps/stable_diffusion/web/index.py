import os
import sys

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"

import gradio as gr
import apps.stable_diffusion.web.utils.global_obj as global_obj
from apps.stable_diffusion.src import args, clear_all
from apps.stable_diffusion.web.utils.gradio_configs import (
    clear_gradio_tmp_imgs_folder,
)
from apps.stable_diffusion.web.ui.utils import get_custom_model_path

# Clear all gradio tmp images from the last session
clear_gradio_tmp_imgs_folder()
# Create the custom model folder if it doesn't already exist
get_custom_model_path().mkdir(parents=True, exist_ok=True)

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
    txt2img_gallery,
    txt2img_sendto_img2img,
    txt2img_sendto_inpaint,
    txt2img_sendto_outpaint,
    txt2img_sendto_upscaler,
    img2img_web,
    img2img_gallery,
    img2img_init_image,
    img2img_sendto_inpaint,
    img2img_sendto_outpaint,
    img2img_sendto_upscaler,
    inpaint_web,
    inpaint_gallery,
    inpaint_init_image,
    inpaint_sendto_img2img,
    inpaint_sendto_outpaint,
    inpaint_sendto_upscaler,
    outpaint_web,
    outpaint_gallery,
    outpaint_init_image,
    outpaint_sendto_img2img,
    outpaint_sendto_inpaint,
    outpaint_sendto_upscaler,
    upscaler_web,
    upscaler_gallery,
    upscaler_init_image,
    upscaler_sendto_img2img,
    upscaler_sendto_inpaint,
    upscaler_sendto_outpaint,
    lora_train_web,
)

# init global sd pipeline and config
global_obj.init()


def register_button_click(button, selectedid, inputs, outputs):
    button.click(
        lambda x: (
            x[0]["name"] if len(x) != 0 else None,
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
        with gr.TabItem(label="Upscaler", id=4):
            upscaler_web.render()
        with gr.TabItem(label="LoRA Training", id=5):
            lora_train_web.render()

    register_button_click(
        txt2img_sendto_img2img,
        1,
        [txt2img_gallery],
        [img2img_init_image, tabs],
    )
    register_button_click(
        txt2img_sendto_inpaint,
        2,
        [txt2img_gallery],
        [inpaint_init_image, tabs],
    )
    register_button_click(
        txt2img_sendto_outpaint,
        3,
        [txt2img_gallery],
        [outpaint_init_image, tabs],
    )
    register_button_click(
        txt2img_sendto_upscaler,
        4,
        [txt2img_gallery],
        [upscaler_init_image, tabs],
    )
    register_button_click(
        img2img_sendto_inpaint,
        2,
        [img2img_gallery],
        [inpaint_init_image, tabs],
    )
    register_button_click(
        img2img_sendto_outpaint,
        3,
        [img2img_gallery],
        [outpaint_init_image, tabs],
    )
    register_button_click(
        img2img_sendto_upscaler,
        4,
        [img2img_gallery],
        [upscaler_init_image, tabs],
    )
    register_button_click(
        inpaint_sendto_img2img,
        1,
        [inpaint_gallery],
        [img2img_init_image, tabs],
    )
    register_button_click(
        inpaint_sendto_outpaint,
        3,
        [inpaint_gallery],
        [outpaint_init_image, tabs],
    )
    register_button_click(
        inpaint_sendto_upscaler,
        4,
        [inpaint_gallery],
        [upscaler_init_image, tabs],
    )
    register_button_click(
        outpaint_sendto_img2img,
        1,
        [outpaint_gallery],
        [img2img_init_image, tabs],
    )
    register_button_click(
        outpaint_sendto_inpaint,
        2,
        [outpaint_gallery],
        [inpaint_init_image, tabs],
    )
    register_button_click(
        outpaint_sendto_upscaler,
        4,
        [outpaint_gallery],
        [upscaler_init_image, tabs],
    )
    register_button_click(
        upscaler_sendto_img2img,
        1,
        [upscaler_gallery],
        [img2img_init_image, tabs],
    )
    register_button_click(
        upscaler_sendto_inpaint,
        2,
        [upscaler_gallery],
        [inpaint_init_image, tabs],
    )
    register_button_click(
        upscaler_sendto_outpaint,
        3,
        [upscaler_gallery],
        [outpaint_init_image, tabs],
    )


sd_web.queue()
sd_web.launch(
    share=args.share,
    inbrowser=True,
    server_name="0.0.0.0",
    server_port=args.server_port,
)
