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

parser = argparse.ArgumentParser(description="SHARK runner.")
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device on which shark_runner runs. options are cpu, cuda, and vulkan",
)
parser.add_argument(
    "--enable_tf32",
    type=bool,
    default=False,
    help="Enables TF32 precision calculations on supported GPUs.",
)
parser.add_argument(
    "--model_config_path",
    help="Directory to where the tuned model config file is located.",
    default=None,
)

parser.add_argument(
    "--num_warmup_iterations",
    type=int,
    default=5,
    help="Run the model for the specified number of warmup iterations.",
)
parser.add_argument(
    "--num_iterations",
    type=int,
    default=100,
    help="Run the model for the specified number of iterations.",
)
parser.add_argument(
    "--onnx_bench",
    default=False,
    action="store_true",
    help="When enabled, pytest bench results will include ONNX benchmark results.",
)
parser.add_argument(
    "--shark_prefix",
    default="latest",
    help="gs://shark_tank/<this_flag>/model_directories",
)
parser.add_argument(
    "--update_tank",
    default=False,
    action="store_true",
    help="When enabled, SHARK downloader will update local shark_tank if local hash is different from latest upstream hash.",
)
parser.add_argument(
    "--local_tank_cache",
    default=None,
    help="Specify where to save downloaded shark_tank artifacts. If this is not set, the default is ~/.local/shark_tank/.",
)

parser.add_argument(
    "--dispatch_benchmarks",
    default=None,
    help='dispatches to return benchamrk data on.  use "All" for all, and None for none.',
)

parser.add_argument(
    "--dispatch_benchmarks_dir",
    default="temp_dispatch_benchmarks",
    help='directory where you want to store dispatch data generated with "--dispatch_benchmarks"',
)

parser.add_argument(
    "--enable_conv_transform",
    default=False,
    action="store_true",
    help="Enables the --iree-flow-enable-conv-nchw-to-nhwc-transform flag.",
)

parser.add_argument(
    "--enable_img2col_transform",
    default=False,
    action="store_true",
    help="Enables the --iree-flow-enable-conv-img2col-transform flag.",
)

parser.add_argument(
    "--use_winograd",
    default=False,
    action="store_true",
    help="Enables the --iree-flow-enable-conv-winograd-transform flag.",
)

shark_args, unknown = parser.parse_known_args()
