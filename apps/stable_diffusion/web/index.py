from multiprocessing import Process, freeze_support
import os
import sys

if sys.platform == "darwin":
    # import before IREE to avoid torch-MLIR library issues
    import torch_mlir

import shutil
import PIL, transformers  # ensures inclusion in pysintaller exe generation
from apps.stable_diffusion.src import args, clear_all
import apps.stable_diffusion.web.utils.global_obj as global_obj

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"
    # import before IREE to avoid MLIR library issues
    import torch_mlir

if args.clear_all:
    clear_all()


def launch_app(address):
    from tkinter import Tk
    import webview

    window = Tk()

    # getting screen width and height of display
    width = window.winfo_screenwidth()
    height = window.winfo_screenheight()
    webview.create_window(
        "SHARK AI Studio", url=address, width=width, height=height
    )
    webview.start(private_mode=False)


if __name__ == "__main__":
    # required to do multiprocessing in a pyinstaller freeze
    freeze_support()
    if args.api or "api" in args.ui.split(","):
        from apps.stable_diffusion.web.ui import (
            txt2img_api,
            img2img_api,
            upscaler_api,
            inpaint_api,
            outpaint_api,
        )
        from fastapi import FastAPI, APIRouter
        import uvicorn

        # init global sd pipeline and config
        global_obj._init()

        app = FastAPI()
        app.add_api_route("/sdapi/v1/txt2img", txt2img_api, methods=["post"])
        app.add_api_route("/sdapi/v1/img2img", img2img_api, methods=["post"])
        app.add_api_route("/sdapi/v1/inpaint", inpaint_api, methods=["post"])
        app.add_api_route("/sdapi/v1/outpaint", outpaint_api, methods=["post"])
        app.add_api_route("/sdapi/v1/upscaler", upscaler_api, methods=["post"])
        app.include_router(APIRouter())
        uvicorn.run(app, host="127.0.0.1", port=args.server_port)
        sys.exit(0)

    # Setup to use shark_tmp for gradio's temporary image files and clear any
    # existing temporary images there if they exist. Then we can import gradio.
    # It has to be in this order or gradio ignores what we've set up.
    from apps.stable_diffusion.web.utils.gradio_configs import (
        config_gradio_tmp_imgs_folder,
    )

    config_gradio_tmp_imgs_folder()
    import gradio as gr

    # Create custom models folders if they don't exist
    from apps.stable_diffusion.web.ui.utils import create_custom_models_folders

    create_custom_models_folders()

    def resource_path(relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        base_path = getattr(
            sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
        )
        return os.path.join(base_path, relative_path)

    dark_theme = resource_path("ui/css/sd_dark_theme.css")

    from apps.stable_diffusion.web.ui import (
        txt2img_web,
        txt2img_custom_model,
        txt2img_hf_model_id,
        txt2img_gallery,
        txt2img_png_info_img,
        txt2img_status,
        txt2img_sendto_img2img,
        txt2img_sendto_inpaint,
        txt2img_sendto_outpaint,
        txt2img_sendto_upscaler,
        img2img_web,
        img2img_custom_model,
        img2img_hf_model_id,
        img2img_gallery,
        img2img_init_image,
        img2img_status,
        img2img_sendto_inpaint,
        img2img_sendto_outpaint,
        img2img_sendto_upscaler,
        inpaint_web,
        inpaint_custom_model,
        inpaint_hf_model_id,
        inpaint_gallery,
        inpaint_init_image,
        inpaint_status,
        inpaint_sendto_img2img,
        inpaint_sendto_outpaint,
        inpaint_sendto_upscaler,
        outpaint_web,
        outpaint_custom_model,
        outpaint_hf_model_id,
        outpaint_gallery,
        outpaint_init_image,
        outpaint_status,
        outpaint_sendto_img2img,
        outpaint_sendto_inpaint,
        outpaint_sendto_upscaler,
        upscaler_web,
        upscaler_custom_model,
        upscaler_hf_model_id,
        upscaler_gallery,
        upscaler_init_image,
        upscaler_status,
        upscaler_sendto_img2img,
        upscaler_sendto_inpaint,
        upscaler_sendto_outpaint,
        lora_train_web,
        model_web,
        hf_models,
        modelmanager_sendto_txt2img,
        modelmanager_sendto_img2img,
        modelmanager_sendto_inpaint,
        modelmanager_sendto_outpaint,
        modelmanager_sendto_upscaler,
        stablelm_chat,
        outputgallery_web,
        outputgallery_tab_select,
        outputgallery_watch,
        outputgallery_filename,
        outputgallery_sendto_txt2img,
        outputgallery_sendto_img2img,
        outputgallery_sendto_inpaint,
        outputgallery_sendto_outpaint,
        outputgallery_sendto_upscaler,
    )

    # init global sd pipeline and config
    global_obj._init()

    def register_button_click(button, selectedid, inputs, outputs):
        button.click(
            lambda x: (
                x[0]["name"] if len(x) != 0 else None,
                gr.Tabs.update(selected=selectedid),
            ),
            inputs,
            outputs,
        )

    def register_modelmanager_button(button, selectedid, inputs, outputs):
        button.click(
            lambda x: (
                "None",
                x,
                gr.Tabs.update(selected=selectedid),
            ),
            inputs,
            outputs,
        )

    def register_outputgallery_button(button, selectedid, inputs, outputs):
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
            with gr.TabItem(label="Upscaler", id=4):
                upscaler_web.render()
            with gr.TabItem(label="Model Manager", id=5):
                model_web.render()
            with gr.TabItem(label="Chat Bot(Experimental)", id=6):
                stablelm_chat.render()
            with gr.TabItem(label="LoRA Training(Experimental)", id=7):
                lora_train_web.render()
            if args.output_gallery:
                with gr.TabItem(label="Output Gallery", id=8) as og_tab:
                    outputgallery_web.render()

                # extra output gallery configuration
                outputgallery_tab_select(og_tab.select)
                outputgallery_watch(
                    [
                        txt2img_status,
                        img2img_status,
                        inpaint_status,
                        outpaint_status,
                        upscaler_status,
                    ]
                )

        # send to buttons
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
        if args.output_gallery:
            register_outputgallery_button(
                outputgallery_sendto_txt2img,
                0,
                [outputgallery_filename],
                [txt2img_png_info_img, tabs],
            )
            register_outputgallery_button(
                outputgallery_sendto_img2img,
                1,
                [outputgallery_filename],
                [img2img_init_image, tabs],
            )
            register_outputgallery_button(
                outputgallery_sendto_inpaint,
                2,
                [outputgallery_filename],
                [inpaint_init_image, tabs],
            )
            register_outputgallery_button(
                outputgallery_sendto_outpaint,
                3,
                [outputgallery_filename],
                [outpaint_init_image, tabs],
            )
            register_outputgallery_button(
                outputgallery_sendto_upscaler,
                4,
                [outputgallery_filename],
                [upscaler_init_image, tabs],
            )
        register_modelmanager_button(
            modelmanager_sendto_txt2img,
            0,
            [hf_models],
            [txt2img_custom_model, txt2img_hf_model_id, tabs],
        )
        register_modelmanager_button(
            modelmanager_sendto_img2img,
            1,
            [hf_models],
            [img2img_custom_model, img2img_hf_model_id, tabs],
        )
        register_modelmanager_button(
            modelmanager_sendto_inpaint,
            2,
            [hf_models],
            [inpaint_custom_model, inpaint_hf_model_id, tabs],
        )
        register_modelmanager_button(
            modelmanager_sendto_outpaint,
            3,
            [hf_models],
            [outpaint_custom_model, outpaint_hf_model_id, tabs],
        )
        register_modelmanager_button(
            modelmanager_sendto_upscaler,
            4,
            [hf_models],
            [upscaler_custom_model, upscaler_hf_model_id, tabs],
        )

    sd_web.queue()
    if args.ui == "app":
        t = Process(
            target=launch_app, args=[f"http://localhost:{args.server_port}"]
        )
        t.start()
    sd_web.launch(
        share=args.share,
        inbrowser=args.ui == "web",
        server_name="0.0.0.0",
        server_port=args.server_port,
    )
