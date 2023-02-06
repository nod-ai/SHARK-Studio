import os
import json
import sys


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_path, relative_path)


def get_json_file(path):
    json_var = []
    loc_json = resource_path(path)
    if os.path.exists(loc_json):
        with open(loc_json, encoding="utf-8") as fopen:
            json_var = json.load(fopen)

    if not json_var:
        print(f"Unable to fetch {path}")

    return json_var


# TODO: This shouldn't be called from here, every time the file imports
# it will run all the global vars.
prompt_examples = get_json_file("resources/prompts.json")
models_db = get_json_file("resources/model_db.json")

# The base_model contains the input configuration for the different
# models and also helps in providing information for the variants.
base_models = get_json_file("resources/base_model.json")

# Contains optimization flags for different models.
opt_flags = get_json_file("resources/opt_flags.json")


# `fetch_and_update_base_model_id` is a resource utility function which
# helps maintaining mapping of the model to run with its base model.
# If `base_model` is "", then this function tries to fetch the base model
# info for the `model_to_run`.
def fetch_and_update_base_model_id(model_to_run, base_model=""):
    path = "resources/variants.json"
    loc_json = resource_path(path)
    data = {model_to_run: base_model}
    json_data = {}
    if os.path.exists(loc_json):
        with open(loc_json, "r", encoding="utf-8") as jsonFile:
            json_data = json.load(jsonFile)
            # Return with base_model's info if base_model is "".
            if base_model == "":
                if model_to_run in json_data:
                    base_model = json_data[model_to_run]
                return base_model
    elif base_model == "":
        return base_model
    # Update JSON data to contain an entry mapping model_to_run with base_model.
    json_data.update(data)
    with open(loc_json, "w", encoding="utf-8") as jsonFile:
        json.dump(json_data, jsonFile)
