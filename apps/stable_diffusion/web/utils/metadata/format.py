# As SHARK has evolved more columns have been added to images_details.csv. However, since
# no version of the CSV has any headers (yet) we don't actually have anything within the
# file that tells us which parameter each column is for. So this is a list of known patterns
# indexed by length which is what we're going to have to use to guess which columns are the
# right ones for the file we're looking at.

# The same ordering is used for JSON, but these do have key names, however they are not very
# human friendly, nor do they match up with the what is written to the .png headers

# So these are functions to try and get something consistent out the raw input from all
# these sources

PARAMS_FORMATS = {
    9: {
        "VARIANT": "Model",
        "SCHEDULER": "Sampler",
        "PROMPT": "Prompt",
        "NEG_PROMPT": "Negative prompt",
        "SEED": "Seed",
        "CFG_SCALE": "CFG scale",
        "PRECISION": "Precision",
        "STEPS": "Steps",
        "OUTPUT": "Filename",
    },
    10: {
        "MODEL": "Model",
        "VARIANT": "Variant",
        "SCHEDULER": "Sampler",
        "PROMPT": "Prompt",
        "NEG_PROMPT": "Negative prompt",
        "SEED": "Seed",
        "CFG_SCALE": "CFG scale",
        "PRECISION": "Precision",
        "STEPS": "Steps",
        "OUTPUT": "Filename",
    },
    12: {
        "VARIANT": "Model",
        "SCHEDULER": "Sampler",
        "PROMPT": "Prompt",
        "NEG_PROMPT": "Negative prompt",
        "SEED": "Seed",
        "CFG_SCALE": "CFG scale",
        "PRECISION": "Precision",
        "STEPS": "Steps",
        "HEIGHT": "Height",
        "WIDTH": "Width",
        "MAX_LENGTH": "Max Length",
        "OUTPUT": "Filename",
    },
}

PARAMS_FORMAT_LONGEST = PARAMS_FORMATS[max(PARAMS_FORMATS.keys())]


def compact(metadata: dict) -> dict:
    # we don't want to alter the original dictionary
    result = dict(metadata)

    # discard the filename because we should already have it
    if result.keys() & {"Filename"}:
        result.pop("Filename")

    # make showing the sizes more compact by using only one line each
    if result.keys() & {"Size-1", "Size-2"}:
        result["Size"] = f"{result.pop('Size-1')}x{result.pop('Size-2')}"
    elif result.keys() & {"Height", "Width"}:
        result["Size"] = f"{result.pop('Height')}x{result.pop('Width')}"

    if result.keys() & {"Hires resize-1", "Hires resize-1"}:
        hires_y = result.pop("Hires resize-1")
        hires_x = result.pop("Hires resize-2")

        if hires_x == 0 and hires_y == 0:
            result["Hires resize"] = "None"
        else:
            result["Hires resize"] = f"{hires_y}x{hires_x}"

    return result


def humanizable(metadata: dict | list[str], includes_filename=True) -> dict:
    lookup_key = len(metadata) + (0 if includes_filename else 1)
    return lookup_key in PARAMS_FORMATS.keys()


def humanize(metadata: dict | list[str], includes_filename=True) -> dict:
    lookup_key = len(metadata) + (0 if includes_filename else 1)

    # For lists we can only work based on the length, we have no other information
    if isinstance(metadata, list):
        if humanizable(metadata, includes_filename):
            return dict(zip(PARAMS_FORMATS[lookup_key].values(), metadata))
        else:
            raise KeyError(
                f"Humanize could not find the format for a parameter list of length {len(metadata)}"
            )

    # For dictionaries we try to use the matching length parameter format if
    # available, otherwise we use the longest. Then we swap keys in the
    # metadata that match keys in the format for the friendlier name that we
    # have set in the format value
    if isinstance(metadata, dict):
        if humanizable(metadata, includes_filename):
            format = PARAMS_FORMATS[lookup_key]
        else:
            format = PARAMS_FORMAT_LONGEST

        return {
            format[key]: value
            for (key, value) in metadata.items()
            if key in format.keys()
        }

    raise TypeError("Can only humanize parameter lists or dictionaries")
