import os
import json
import sys


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_path, relative_path)


prompt_examples = []
prompts_loc = resource_path("resources/prompts.json")
if os.path.exists(prompts_loc):
    with open(prompts_loc, encoding="utf-8") as fopen:
        prompt_examples = json.load(fopen)

if not prompt_examples:
    print("Unable to fetch prompt examples.")


models_db = []
models_loc = resource_path("resources/model_db.json")
if os.path.exists(models_loc):
    with open(models_loc, encoding="utf-8") as fopen:
        models_db = json.load(fopen)

if len(models_db) != 3:
    sys.exit("Error: Unable to load models database.")

models_config = []
models_config_loc = resource_path("resources/model_config.json")
if os.path.exists(models_config_loc):
    with open(models_config_loc, encoding="utf-8") as fopen:
        models_config = json.load(fopen)

if not models_config:
    sys.exit("Error: Unable to load models config.")
