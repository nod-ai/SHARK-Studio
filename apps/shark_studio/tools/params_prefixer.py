from apps.shark_studio.modules.ckpt_processing import save_irpa
import argparse
import safetensors

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    default="",
    help="input safetensors/irpa",
)
parser.add_argument(
    "--prefix",
    type=str,
    default="",
    help="prefix to add to all the keys in the irpa",
)
parser.add_argument(
    "--replace",
    type=str,
    default=None,
    help="prefix to be removed"
)
args = parser.parse_args()
output_file = save_irpa(args.input, args.prefix, args.replace)
print("saved irpa to", output_file, "with prefix", args.prefix)
