import sys

import gradio as gr

from modules import (
    shared_cmd_options,
    shared_gradio,
    options,
    shared_items,
    sd_models_types,
)
from modules.paths_internal import (
    models_path,
    script_path,
    data_path,
    sd_configs_path,
    sd_default_config,
    sd_model_file,
    default_sd_model_file,
    extensions_dir,
    extensions_builtin_dir,
)  # noqa: F401
from modules import util

cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

parallel_processing_allowed = True
styles_filename = cmd_opts.styles_file
config_filename = cmd_opts.ui_settings_file

demo = None

device = None

weight_load_location = None

state = None

prompt_styles = None

options_templates = None
opts = None
restricted_opts = None

sd_model: sd_models_types.WebuiSdModel = None

settings_components = None
"""assinged from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""

tab_names = []

sd_upscalers = []

clip_model = None

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm = None

mem_mon = None

reload_gradio_theme = shared_gradio.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
