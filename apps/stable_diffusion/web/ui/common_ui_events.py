import gradio as gr

from apps.stable_diffusion.web.ui.utils import (
    HSLHue,
    hsl_color,
    get_lora_metadata,
)


# Answers HTML to show the most frequent tags used when a LoRA was trained,
# taken from the metadata of its .safetensors file.
def lora_changed(lora_file):
    # tag frequency percentage, that gets maximum amount of the staring hue
    TAG_COLOR_THRESHOLD = 0.55
    # tag frequency percentage, above which a tag is displayed
    TAG_DISPLAY_THRESHOLD = 0.65
    # template for the html used to display a tag
    TAG_HTML_TEMPLATE = '<span class="lora-tag" style="border: 1px solid {color};">{tag}</span>'

    if lora_file == "None":
        return ["<div><i>No LoRA selected</i></div>"]
    elif not lora_file.lower().endswith(".safetensors"):
        return [
            "<div><i>Only metadata queries for .safetensors files are currently supported</i></div>"
        ]
    else:
        metadata = get_lora_metadata(lora_file)
        if metadata:
            frequencies = metadata["frequencies"]
            return [
                "".join(
                    [
                        f'<div class="lora-model">Trained against weights in: {metadata["model"]}</div>'
                    ]
                    + [
                        TAG_HTML_TEMPLATE.format(
                            color=hsl_color(
                                (tag[1] - TAG_COLOR_THRESHOLD)
                                / (1 - TAG_COLOR_THRESHOLD),
                                start=HSLHue.RED,
                                end=HSLHue.GREEN,
                            ),
                            tag=tag[0],
                        )
                        for tag in frequencies
                        if tag[1] > TAG_DISPLAY_THRESHOLD
                    ],
                )
            ]
        elif metadata is None:
            return [
                "<div><i>This LoRA does not publish tag frequency metadata</i></div>"
            ]
        else:
            return [
                "<div><i>This LoRA has empty tag frequency metadata, or we could not parse it</i></div>"
            ]
