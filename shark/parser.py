# Copyright 2020 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


parser = argparse.ArgumentParser(description='SHARK runner.')
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device on which shark_runner runs. options are cpu, gpu, and vulkan")
parser.add_argument(
    "--repro_dir",
    help=
    "Directory to which module files will be saved for reproduction or debugging.",
    type=dir_path,
    default="/tmp/")
parser.add_argument(
    "--save_mlir",
    default=False,
    action="store_true",
    help="Saves input MLIR module to /tmp/ directory.")
parser.add_argument(
    "--save_vmfb",
    default=False,
    action="store_true",
    help="Saves iree .vmfb module to /tmp/ directory.")
parser.add_argument(
    "--num_warmup_iterations",
    type=int,
    default=2,
    help="Run the model for the specified number of warmup iterations.")
parser.add_argument(
    "--num_iterations",
    type=int,
    default=1,
    help="Run the model for the specified number of iterations.")

shark_args = parser.parse_args()
