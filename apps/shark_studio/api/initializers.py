import importlib
import os
import signal
import sys
import warnings
import json
from threading import Thread

from apps.shark_studio.modules.timer import startup_timer
from apps.shark_studio.web.utils.tmp_configs import (
        config_tmp,
        clear_tmp_mlir,
        clear_tmp_imgs,
    )


def imports():
    import torch  # noqa: F401

    startup_timer.record("import torch")
    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, module="torch"
    )
    warnings.filterwarnings(
        action="ignore", category=UserWarning, module="torchvision"
    )
    warnings.filterwarnings(
        action="ignore", category=UserWarning, message='.*is deprecated, please use.*', module="*torch*"
    )

    import gradio  # noqa: F401

    startup_timer.record("import gradio")

    import apps.shark_studio.web.utils.globals as global_obj

    global_obj._init()
    startup_timer.record("initialize globals")

    from apps.shark_studio.modules import (
        img_processing,
    )  # noqa: F401

    startup_timer.record("other imports")


def initialize():
    configure_sigint_handler()
    # Setup to use shark_tmp for gradio's temporary image files and clear any
    # existing temporary images there if they exist. Then we can import gradio.
    # It has to be in this order or gradio ignores what we've set up.

    config_tmp()
    clear_tmp_mlir()
    clear_tmp_imgs()

    from apps.shark_studio.web.utils.file_utils import (
        create_checkpoint_folders,
    )
    # Create custom models folders if they don't exist
    create_checkpoint_folders()

    import gradio as gr

    # initialize_rest(reload_script_modules=False)


def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    # Keep this for adding reload options to the webUI.


def dumpstacks():
    import threading
    import traceback

    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append(f"\n# Thread: {id2name.get(threadId, '')}({threadId})")
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append(f"""File: "{filename}", line {lineno}, in {name}""")
            if line:
                code.append("  " + line.strip())

    print("\n".join(code))


def configure_sigint_handler():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f"Interrupted with signal {sig} in {frame}")

        dumpstacks()

        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
