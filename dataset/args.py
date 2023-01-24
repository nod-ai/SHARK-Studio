import argparse

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

##############################################################################
### Dataset Annotator flags
##############################################################################

p.add_argument(
    "--gs_url",
    type=str,
    required=True,
    help="URL to datasets in GS bucket",
)

p.add_argument(
    "--share",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="flag for generating a public URL",
)

p.add_argument(
    "--server_port",
    type=int,
    default=8080,
    help="flag for setting server port",
)

##############################################################################

args = p.parse_args()
