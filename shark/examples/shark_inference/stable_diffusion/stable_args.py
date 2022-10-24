import argparse

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

p.add_argument(
    "--prompt",
    type=str,
    default="a photograph of an astronaut riding a horse",
    help="the text to generate image of.",
)
p.add_argument(
    "--device", type=str, default="cpu", help="device to run the model."
)
p.add_argument(
    "--steps",
    type=int,
    default=10,
    help="the no. of steps to do the sampling.",
)
p.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed to use.",
)
p.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="the value to be used for guidance scaling.",
)

p.add_argument(
    "--import_mlir",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="imports the model from torch module to shark_module otherwise downloads the model from shark_tank.",
)

p.add_argument(
    "--precision", type=str, default="fp32", help="precision to run the model."
)

p.add_argument(
    "--max_length",
    type=int,
    default=77,
    help="max length of the tokenizer output.",
)

p.add_argument(
    "--load_vmfb",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="attempts to load the model from a precompiled flatbuffer and compiles + saves it if not found.",
)

p.add_argument(
    "--save_vmfb",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="saves the compiled flatbuffer to the local directory",
)


args = p.parse_args()
