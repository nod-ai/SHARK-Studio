import json
import os
from PIL import Image
from .png_metadata import parse_generation_parameters
from .exif_metadata import has_exif, parse_exif
from .csv_metadata import has_csv, parse_csv
from .format import compact, humanize


def displayable_metadata(image_filename: str) -> dict:
    pil_image = Image.open(image_filename)

    # we have PNG generation parameters (preferred, as it's what the txt2img dropzone reads,
    # and we go via that for SendTo, and is directly tied to the image)
    if "parameters" in pil_image.info:
        return {
            "source": "png",
            "parameters": compact(
                parse_generation_parameters(pil_image.info["parameters"])
            ),
        }

    # we have a matching json file (next most likely to be accurate when it's there)
    json_path = os.path.splitext(image_filename)[0] + ".json"
    if os.path.isfile(json_path):
        with open(json_path) as params_file:
            return {
                "source": "json",
                "parameters": compact(
                    humanize(json.load(params_file), includes_filename=False)
                ),
            }

    # we have a CSV file so try that (can be different shapes, and it usually has no
    # headers/param names so of the things we we *know* have parameters, it's the
    # last resort)
    if has_csv(image_filename):
        params = parse_csv(image_filename)
        if params:  # we might not have found the filename in the csv
            return {
                "source": "csv",
                "parameters": compact(params),  # already humanized
            }

    # EXIF data, probably a .jpeg, may well not include parameters, but at least it's *something*
    if has_exif(image_filename):
        return {"source": "exif", "parameters": parse_exif(pil_image)}

    # we've got nothing
    return None
