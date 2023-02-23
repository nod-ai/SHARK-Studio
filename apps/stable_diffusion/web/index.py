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
    img2img_web,
    inpaint_web,
    outpaint_web,
)

sd_web = gr.TabbedInterface(
    [txt2img_web, img2img_web, inpaint_web, outpaint_web],
    ["Text-to-Image", "Image-to-Image", "Inpainting", "Outpainting"],
    css=dark_theme,
)

sd_web.queue()
sd_web.launch(
    share=args.share,
    inbrowser=True,
    server_name="0.0.0.0",
    server_port=args.server_port,
)
