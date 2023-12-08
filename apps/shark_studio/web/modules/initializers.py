import importlib
import logging
import os
import signal
import sys
import re
import warnings
import json
from threading import Thread

from modules.timer import startup_timer


def imports():
    import torch  # noqa: F401
    startup_timer.record("import torch")
    warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="torch")
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")

    import gradio  # noqa: F401
    startup_timer.record("import gradio")

    from apps.shark_studio.modules import shared_init
    shared_init.initialize()
    startup_timer.record("initialize shared")

    from apps.shark_studio.modules import processing, gradio_extensons, ui  # noqa: F401
    startup_timer.record("other imports")

def initialize():
    configure_sigint_handler()
    configure_opts_onchange()

    from apps.shark_studio.modules import modelloader
    modelloader.cleanup_models()

    from apps.shark_studio.modules import sd_models
    sd_models.setup_model()
    startup_timer.record("setup SD model")

    #from apps.shark_studio.modules.shared_cmd_options import cmd_opts

    #from apps.shark_studio.modules import codeformer_model
    #warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
    #codeformer_model.setup_model(cmd_opts.codeformer_models_path)
    #startup_timer.record("setup codeformer")

    #from apps.shark_studio.modules import gfpgan_model
    #gfpgan_model.setup_model(cmd_opts.gfpgan_models_path)
    #startup_timer.record("setup gfpgan")

    initialize_rest(reload_script_modules=False)

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
        print(f'Interrupted with signal {sig} in {frame}')

        dumpstacks()

        os._exit(0)

    if not os.environ.get("COVERAGE_RUN"):
        # Don't install the immediate-quit handler when running under coverage,
        # as then the coverage report won't be generated.
    signal.signal(signal.SIGINT, sigint_handler)


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


def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    from apps.shark_studio.modules.shared_cmd_options import cmd_opts

    from apps.shark_studio.modules import sd_samplers
    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    restore_config_state_file()
    startup_timer.record("restore config state file")

    from apps.shark_studio.modules import sd_models
    sd_models.list_models()
    startup_timer.record("list SD models")

    with startup_timer.subcategory("load scripts"):
        scripts.load_scripts()

    if reload_script_modules:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    from apps.shark_studio.modules import sd_vae
    sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    # from apps.shark_studio.modules import textual_inversion
    # textual_inversion.textual_inversion.list_textual_inversion_templates()
    # startup_timer.record("refresh textual inversion templates")

    from apps.shark_studio.modules import sd_unet
    sd_unet.list_unets()
    startup_timer.record("scripts list_unets")

    def load_model():
        """
        Accesses shared.sd_model property to load model.
        """

        shared.sd_model  # noqa: B018

    Thread(target=load_model).start()
