from multiprocessing import freeze_support
import os
import sys
import logging
import warnings
import apps.stable_diffusion.web.utils.app as app

if sys.platform == "darwin":
    # import before IREE to avoid torch-MLIR library issues
    import torch_mlir

import shutil
import PIL, transformers, sentencepiece  # ensures inclusion in pysintaller exe generation
from apps.stable_diffusion.src import args, clear_all
import apps.stable_diffusion.web.utils.global_obj as global_obj

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"
    # import before IREE to avoid MLIR library issues
    import torch_mlir

if args.clear_all:
    clear_all()


# This function is intended to clean up MEI folders
def cleanup_mei_folders():

    # Determine the operating system
    if sys.platform.startswith('win'):
        temp_dir = os.path.join(os.environ['LOCALAPPDATA'], 'Temp')

    # For potential extension to support Linux or macOS systems:
    # NOTE: Before enabling, ensure compatibility and testing.
    # elif sys.platform.startswith('linux') or sys.platform == 'darwin':
    #    temp_dir = '/tmp'

    else:
        warnings.warn("Temporary files weren't deleted due to an unsupported OS; program functionality is unaffected.")
        return

    prefix = '_MEI'

    # Iterate through the items in the temporary directory
    for item in os.listdir(temp_dir):
        if item.startswith(prefix):
            path = os.path.join(temp_dir, item)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    # required to do multiprocessing in a pyinstaller freeze
    freeze_support()
    if args.api or "api" in args.ui.split(","):
        from apps.stable_diffusion.web.ui import (
            llm_chat_api,
        )
        from apps.stable_diffusion.web.api import sdapi

        from fastapi import FastAPI, APIRouter
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn

        # init global sd pipeline and config
        global_obj._init()

        api = FastAPI()
        api.mount("/sdapi/", sdapi)

        # chat APIs needed for compatibility with multiple extensions using OpenAI API
        api.add_api_route(
            "/v1/chat/completions", llm_chat_api, methods=["post"]
        )
        api.add_api_route("/v1/completions", llm_chat_api, methods=["post"])
        api.add_api_route("/chat/completions", llm_chat_api, methods=["post"])
        api.add_api_route("/completions", llm_chat_api, methods=["post"])
        api.add_api_route(
            "/v1/engines/codegen/completions", llm_chat_api, methods=["post"]
        )
        api.include_router(APIRouter())

        # deal with CORS requests if CORS accept origins are set
        if args.api_accept_origin:
            print(
                f"API Configured for CORS. Accepting origins: { args.api_accept_origin }"
            )
            api.add_middleware(
                CORSMiddleware,
                allow_origins=args.api_accept_origin,
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )
        else:
            print("API not configured for CORS")

        uvicorn.run(api, host="0.0.0.0", port=args.server_port)
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
    from apps.stable_diffusion.web.ui.utils import (
        create_custom_models_folders,
        nodicon_loc,
    )

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
        txt2img_gallery,
        txt2img_png_info_img,
        txt2img_status,
        txt2img_sendto_img2img,
        txt2img_sendto_inpaint,
        txt2img_sendto_outpaint,
        txt2img_sendto_upscaler,
        # h2ogpt_upload,
        # h2ogpt_web,
        img2img_web,
        img2img_custom_model,
        img2img_gallery,
        img2img_init_image,
        img2img_status,
        img2img_sendto_inpaint,
        img2img_sendto_outpaint,
        img2img_sendto_upscaler,
        inpaint_web,
        inpaint_custom_model,
        inpaint_gallery,
        inpaint_init_image,
        inpaint_status,
        inpaint_sendto_img2img,
        inpaint_sendto_outpaint,
        inpaint_sendto_upscaler,
        outpaint_web,
        outpaint_custom_model,
        outpaint_gallery,
        outpaint_init_image,
        outpaint_status,
        outpaint_sendto_img2img,
        outpaint_sendto_inpaint,
        outpaint_sendto_upscaler,
        upscaler_web,
        upscaler_custom_model,
        upscaler_gallery,
        upscaler_init_image,
        upscaler_status,
        upscaler_sendto_img2img,
        upscaler_sendto_inpaint,
        upscaler_sendto_outpaint,
        #  lora_train_web,
        #  model_web,
        #  model_config_web,
        hf_models,
        modelmanager_sendto_txt2img,
        modelmanager_sendto_img2img,
        modelmanager_sendto_inpaint,
        modelmanager_sendto_outpaint,
        modelmanager_sendto_upscaler,
        stablelm_chat,
        minigpt4_web,
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
        css=dark_theme, analytics_enabled=False, title="SHARK AI Studio"
    ) as sd_web:
        with gr.Tabs() as tabs:
            # NOTE: If adding, removing, or re-ordering tabs, make sure that they
            # have a unique id that doesn't clash with any of the other tabs,
            # and that the order in the code here is the order they should
            # appear in the ui, as the id value doesn't determine the order.

            # Where possible, avoid changing the id of any tab that is the
            # destination of one of the 'send to' buttons. If you do have to change
            # that id, make sure you update the relevant register_button_click calls
            # further down with the new id.
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
            if args.output_gallery:
                with gr.TabItem(label="Output Gallery", id=5) as og_tab:
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
            #  with gr.TabItem(label="Model Manager", id=6):
            #      model_web.render()
            #  with gr.TabItem(label="LoRA Training (Experimental)", id=7):
            #      lora_train_web.render()
            with gr.TabItem(label="Chat Bot", id=8):
                stablelm_chat.render()
            #  with gr.TabItem(
            #      label="Generate Sharding Config (Experimental)", id=9
            #  ):
            #      model_config_web.render()
            with gr.TabItem(label="MultiModal (Experimental)", id=10):
                minigpt4_web.render()
            # with gr.TabItem(label="DocuChat Upload", id=11):
            #     h2ogpt_upload.render()
            # with gr.TabItem(label="DocuChat(Experimental)", id=12):
            #     h2ogpt_web.render()

            actual_port = app.usable_port()
            if actual_port != args.server_port:
                sd_web.load(
                    fn=lambda: gr.Info(
                        f"Port {args.server_port} is in use by another application. "
                        f"Shark is running on port {actual_port} instead."
                    )
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
            [txt2img_custom_model, tabs],
        )
        register_modelmanager_button(
            modelmanager_sendto_img2img,
            1,
            [hf_models],
            [img2img_custom_model, tabs],
        )
        register_modelmanager_button(
            modelmanager_sendto_inpaint,
            2,
            [hf_models],
            [inpaint_custom_model, tabs],
        )
        register_modelmanager_button(
            modelmanager_sendto_outpaint,
            3,
            [hf_models],
            [outpaint_custom_model, tabs],
        )
        register_modelmanager_button(
            modelmanager_sendto_upscaler,
            4,
            [hf_models],
            [upscaler_custom_model, tabs],
        )

    sd_web.queue()
    sd_web.launch(
        share=args.share,
        inbrowser=not app.launch(actual_port),
        server_name="0.0.0.0",
        server_port=actual_port,
        favicon_path=nodicon_loc,
    )
    cleanup_mei_folders()
