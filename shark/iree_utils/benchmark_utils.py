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

from shark.iree_utils._common import run_cmd, iree_device_map
from shark.iree_utils.cpu_utils import get_cpu_count
import numpy as np
import os
import re
import platform

UNIT_TO_SECOND_MAP = {"us": 1e-6, "ms": 0.001, "s": 1}


def tensor_to_type_str(input_tensors: tuple, mlir_dialect: str):
    """
    Input: A tuple of input tensors i.e tuple(torch.tensor)
    Output: list of string that represent mlir types (i.e 1x24xf64)
    # TODO: Support more than floats, and ints
    """
    list_of_type = []
    for input_tensor in input_tensors:
        type_string = "x".join([str(dim) for dim in input_tensor.shape])
        if mlir_dialect in ["linalg", "tosa"]:
            dtype_string = str(input_tensor.dtype).replace("torch.", "")
        elif mlir_dialect in ["mhlo", "tflite"]:
            dtype = input_tensor.dtype
            try:
                dtype_string = re.findall("'[^\"]*'", str(dtype))[0].replace(
                    "'", ""
                )
            except IndexError:
                dtype_string = str(dtype)
        regex_split = re.compile("([a-zA-Z]+)([0-9]+)")
        match = regex_split.match(dtype_string)
        mlir_type_string = str(match.group(1)[0]) + str(match.group(2))
        type_string += f"x{mlir_type_string}"
        list_of_type.append(type_string)
    return list_of_type


def build_benchmark_args(
    input_file: str,
    device: str,
    input_tensors: tuple,
    mlir_dialect: str,
    training=False,
):
    """
    Inputs: input_file leading to vmfb, input_tensor to function, target device,
    and whether it is training or not.
    Outputs: string that execute benchmark-module on target model.
    """
    path = os.path.join(os.environ["VIRTUAL_ENV"], "bin")
    if platform.system() == "Windows":
        benchmarker_path = os.path.join(path, "iree-benchmark-module.exe")
        time_extractor = None
    else:
        benchmarker_path = os.path.join(path, "iree-benchmark-module")
        time_extractor = "| awk 'END{{print $2 $3}}'"
    benchmark_cl = [benchmarker_path, f"--module={input_file}"]
    # TODO: The function named can be passed as one of the args.
    fn_name = "forward"
    if training == True:
        # TODO: Replace name of train with actual train fn name.
        fn_name = "train"
    benchmark_cl.append(f"--function={fn_name}")
    benchmark_cl.append(f"--device={iree_device_map(device)}")
    mlir_input_types = tensor_to_type_str(input_tensors, mlir_dialect)
    for mlir_input in mlir_input_types:
        benchmark_cl.append(f"--input={mlir_input}")
    if device == "cpu":
        num_cpus = get_cpu_count()
        if num_cpus is not None:
            benchmark_cl.append(f"--task_topology_max_group_count={num_cpus}")
    # if time_extractor:
    #    benchmark_cl.append(time_extractor)
    benchmark_cl.append(f"--print_statistics=true")
    return benchmark_cl


def build_benchmark_args_non_tensor_input(
    input_file: str,
    device: str,
    inputs: tuple,
    mlir_dialect: str,
    function_name: str,
):
    """
    Inputs: input_file leading to vmfb, input_tensor to function, target device,
    and whether it is training or not.
    Outputs: string that execute benchmark-module on target model.
    """
    path = os.path.join(os.environ["VIRTUAL_ENV"], "bin")
    if platform.system() == "Windows":
        benchmarker_path = os.path.join(path, "iree-benchmark-module.exe")
        time_extractor = None
    else:
        benchmarker_path = os.path.join(path, "iree-benchmark-module")
        time_extractor = "| awk 'END{{print $2 $3}}'"
    benchmark_cl = [benchmarker_path, f"--module={input_file}"]
    # TODO: The function named can be passed as one of the args.
    if function_name:
        benchmark_cl.append(f"--function={function_name}")
    benchmark_cl.append(f"--device={iree_device_map(device)}")
    for input in inputs:
        benchmark_cl.append(f"--input={input}")
    if platform.system() != "Windows":
        time_extractor = "| awk 'END{{print $2 $3}}'"
        benchmark_cl.append(time_extractor)
    return benchmark_cl


def run_benchmark_module(benchmark_cl):
    """
    Run benchmark command, extract result and return iteration/seconds, host
    peak memory, and device peak memory.

    # TODO: Add an example of the benchmark command.
    Input: benchmark command.
    """
    benchmark_path = benchmark_cl[0]
    assert os.path.exists(
        benchmark_path
    ), "Cannot find iree_benchmark_module, Please contact SHARK maintainer on discord."
    bench_stdout, bench_stderr = run_cmd(" ".join(benchmark_cl))
    try:
        regex_split = re.compile("(\d+[.]*\d*)(  *)([a-zA-Z]+)")
        match = regex_split.search(bench_stdout)
        time_ms = float(match.group(1))
        unit = match.group(3)
    except AttributeError:
        regex_split = re.compile("(\d+[.]*\d*)([a-zA-Z]+)")
        match = regex_split.search(bench_stdout)
        time_ms = float(match.group(1))
        unit = match.group(2)
    iter_per_second = 1.0 / (time_ms * 0.001)

    # Extract peak memory.
    host_regex = re.compile(r".*HOST_LOCAL:\s*([0-9]+)B peak")
    host_peak_b = int(host_regex.search(bench_stderr).group(1))
    device_regex = re.compile(r".*DEVICE_LOCAL:\s*([0-9]+)B peak")
    device_peak_b = int(device_regex.search(bench_stderr).group(1))
    return iter_per_second, host_peak_b, device_peak_b
