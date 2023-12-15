from multiprocessing import Process, freeze_support
import os
import time
import sys
import logging
import apps.shark_studio.api.initializers as initialize


from apps.shark_studio.modules import timer

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.imports()

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"
    # import before IREE to avoid MLIR library issues
    import torch_mlir


def create_api(app):
    from apps.shark_studio.api.compat import ApiCompat
    from modules.call_queue import queue_lock

    api = ApiCompat(app, queue_lock)
    return api


def api_only():
    from fastapi import FastAPI
    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts

    initialize.initialize()

    app = FastAPI()
    initialize.setup_middleware(app)
    api = create_api(app)

    # from modules import script_callbacks
    # script_callbacks.before_ui_callback()
    # script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(
        server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1",
        port=cmd_opts.port if cmd_opts.port else 8080,
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
    )


def launch_webui(address):
    from tkinter import Tk
    import webview

    window = Tk()

    # get screen width and height of display and make it more reasonably
    # sized as we aren't making it full-screen or maximized
    width = int(window.winfo_screenwidth() * 0.81)
    height = int(window.winfo_screenheight() * 0.91)
    webview.create_window(
        "SHARK AI Studio",
        url=address,
        width=width,
        height=height,
        text_select=True,
    )
    webview.start(private_mode=False, storage_path=os.getcwd())


def webui():
    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts

    launch_api = cmd_opts.api
    initialize.initialize()

    from ui.chat import chat_element
    from ui.sd import sd_element
    from ui.outputgallery import outputgallery_element

    # required to do multiprocessing in a pyinstaller freeze
    freeze_support()

    # if args.api or "api" in args.ui.split(","):
    #     from apps.shark_studio.api.llm import (
    #         chat,
    #     )
    #     from apps.shark_studio.web.api import sdapi
    #
    #     from fastapi import FastAPI, APIRouter
    #     from fastapi.middleware.cors import CORSMiddleware
    #     import uvicorn
    #
    #     # init global sd pipeline and config
    #     global_obj._init()
    #
    #     api = FastAPI()
    #     api.mount("/sdapi/", sdapi)
    #
    #     # chat APIs needed for compatibility with multiple extensions using OpenAI API
    #     api.add_api_route(
    #         "/v1/chat/completions", llm_chat_api, methods=["post"]
    #     )
    #     api.add_api_route("/v1/completions", llm_chat_api, methods=["post"])
    #     api.add_api_route("/chat/completions", llm_chat_api, methods=["post"])
    #     api.add_api_route("/completions", llm_chat_api, methods=["post"])
    #     api.add_api_route(
    #         "/v1/engines/codegen/completions", llm_chat_api, methods=["post"]
    #     )
    #     api.include_router(APIRouter())
    #
    #     # deal with CORS requests if CORS accept origins are set
    #     if args.api_accept_origin:
    #         print(
    #             f"API Configured for CORS. Accepting origins: { args.api_accept_origin }"
    #         )
    #         api.add_middleware(
    #             CORSMiddleware,
    #             allow_origins=args.api_accept_origin,
    #             allow_methods=["GET", "POST"],
    #             allow_headers=["*"],
    #         )
    #     else:
    #         print("API not configured for CORS")
    #
    #     uvicorn.run(api, host="0.0.0.0", port=args.server_port)
    #     sys.exit(0)
    # Setup to use shark_tmp for gradio's temporary image files and clear any
    # existing temporary images there if they exist. Then we can import gradio.
    # It has to be in this order or gradio ignores what we've set up.
    from apps.shark_studio.web.utils.tmp_configs import (
        config_tmp,
        clear_tmp_mlir,
        clear_tmp_imgs,
    )
    from apps.shark_studio.web.utils.file_utils import (
        create_checkpoint_folders,
    )

    import gradio as gr

    config_tmp()
    clear_tmp_mlir()
    clear_tmp_imgs()

    # Create custom models folders if they don't exist
    create_checkpoint_folders()

    def resource_path(relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        base_path = getattr(
            sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
        )
        return os.path.join(base_path, relative_path)

    dark_theme = resource_path("ui/css/sd_dark_theme.css")

    # from apps.shark_studio.web.ui import load_ui_from_script

    def register_button_click(button, selectedid, inputs, outputs):
        button.click(
            lambda x: (
                x[0]["name"] if len(x) != 0 else None,
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
        css=dark_theme, analytics_enabled=False, title="Shark Studio 2.0 Beta"
    ) as studio_web:
        with gr.Tabs() as tabs:
            # NOTE: If adding, removing, or re-ordering tabs, make sure that they
            # have a unique id that doesn't clash with any of the other tabs,
            # and that the order in the code here is the order they should
            # appear in the ui, as the id value doesn't determine the order.

            # Where possible, avoid changing the id of any tab that is the
            # destination of one of the 'send to' buttons. If you do have to change
            # that id, make sure you update the relevant register_button_click calls
            # further down with the new id.
            with gr.TabItem(label="Stable Diffusion", id=0):
                sd_element.render()
            with gr.TabItem(label="Output Gallery", id=1):
                outputgallery_element.render()
            with gr.TabItem(label="Chat Bot", id=2):
                chat_element.render()

    studio_web.queue()

    # if args.ui == "app":
    #    t = Process(
    #        target=launch_app, args=[f"http://localhost:{args.server_port}"]
    #    )
    #    t.start()
    studio_web.launch(
        share=True,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=11911,  # args.server_port,
    )


if __name__ == "__main__":
    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts

    if cmd_opts.webui == False:
        api_only()
    else:
        webui()
