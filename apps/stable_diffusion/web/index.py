import os
import sys


if "AMD_ENABLE_LLPC" not in os.environ:
    os.environ["AMD_ENABLE_LLPC"] = "1"

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"


import gradio as gr
from apps.stable_diffusion.src import args
from apps.stable_diffusion.web.ui import txt2img_web, img2img_web
from apps.stable_diffusion.web.utils.gradio_configs import (
    clear_gradio_tmp_imgs_folder,
)

# clear all gradio tmp images from the last session
clear_gradio_tmp_imgs_folder()

sd_web = gr.TabbedInterface(
    [txt2img_web, img2img_web], ["Text-to-Image", "Image-to-Image"]
)
sd_web.queue()
sd_web.launch(
    share=args.share,
    inbrowser=True,
    server_name="0.0.0.0",
    server_port=args.server_port,
)
