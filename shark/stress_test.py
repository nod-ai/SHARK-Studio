# Copyright 2022 The Nod Team. All rights reserved.
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

from iree.runtime import query_available_drivers, get_driver
from shark.shark_downloader import download_model
from shark.shark_inference import SharkInference
from typing import List, Optional, Tuple
import numpy as np
import argparse
from shark.iree_utils._common import _IREE_DEVICE_MAP
import multiprocessing
from shark.shark_runner import supported_dialects
import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
import time
import numpy as np

IREE_TO_SHARK_DRIVER_MAP = {v: k for k, v in _IREE_DEVICE_MAP.items()}


def stress_test_compiled_model(
    shark_module_path: str,
    function_name: str,
    device: str,
    inputs: List[np.ndarray],
    golden_out: List[np.ndarray],
    batch_size: int,
    max_iterations: int,
    max_duration_seconds: float,
    inference_timeout_seconds: float,
    tolerance_nulp: int,
    stress_test_index: int,
):
    logging.info(
        f"Running stress test {stress_test_index} on device {device}."
    )
    shark_module = SharkInference(
        mlir_module=bytes(), function_name=function_name, device=device
    )
    shark_module.load_module(shark_module_path)
    input_batches = [np.repeat(arr, batch_size, axis=0) for arr in inputs]
    golden_output_batches = np.repeat(golden_out, batch_size, axis=0)
    report_interval_seconds = 10
    start_time = time.time()
    previous_report_time = start_time
    executor = ThreadPoolExecutor(1)
    first_iteration_output = None
    for i in range(max_iterations):
        inference_task = executor.submit(shark_module.forward, input_batches)
        output = inference_task.result(inference_timeout_seconds)
        if first_iteration_output is None:
            np.testing.assert_array_almost_equal_nulp(
                golden_output_batches, output, nulp=tolerance_nulp
            )
            first_iteration_output = output
        else:
            np.testing.assert_array_equal(output, first_iteration_output)
        current_time = time.time()
        if report_interval_seconds < current_time - previous_report_time:
            logging.info(
                f"Stress test {stress_test_index} on device "
                f"{device} at iteration {i+1}"
            )
            previous_report_time = current_time
        if max_duration_seconds < current_time - start_time:
            return
    logging.info(f"Stress test {stress_test_index} on device {device} done.")


def get_device_type(device_name: str):
    return device_name.split("://", 1)[0]


def get_device_types(device_names: str):
    return [get_device_type(device_name) for device_name in device_names]


def query_devices(device_types: Optional[List[str]] = None) -> List[str]:
    devices = []
    if device_types is None:
        device_types = [
            IREE_TO_SHARK_DRIVER_MAP[name]
            for name in query_available_drivers()
            if name in IREE_TO_SHARK_DRIVER_MAP
        ]
    for device_type in device_types:
        driver = get_driver(_IREE_DEVICE_MAP[device_type])
        device_infos = driver.query_available_devices()
        for device_info in device_infos:
            uri_path = (
                device_info["path"]
                if device_info["path"] != ""
                else str(device_info["device_id"])
            )
            device_uri = f"{device_type}://{uri_path}"
            devices.append(device_uri)
    return devices


def compile_stress_test_module(
    device_types: List[str], mlir_model: str, func_name: str, mlir_dialect: str
) -> List[str]:
    shark_module_paths = []
    for device_type in device_types:
        logging.info(
            f"Compiling stress test model for device type {device_type}."
        )
        shark_module = SharkInference(
            mlir_model,
            func_name,
            mlir_dialect=mlir_dialect,
            device=device_type,
        )
        shark_module_paths.append(shark_module.save_module())
    return shark_module_paths


def stress_test(
    model_name: str,
    dynamic_model: bool = False,
    device_types: Optional[List[str]] = None,
    device_names: Optional[List[str]] = None,
    batch_size: int = 1,
    max_iterations: int = 10**7,
    max_duration_seconds: float = 3600,
    inference_timeout_seconds: float = 60,
    mlir_dialect: str = "linalg",
    frontend: str = "torch",
    oversubscription_factor: int = 1,
    tolerance_nulp: int = 50000,
):
    logging.info(f"Downloading stress test model {model_name}.")
    mlir_model, func_name, inputs, golden_out = download_model(
        model_name=model_name, dynamic=dynamic_model, frontend=frontend
    )

    if device_names is None or device_types is not None:
        device_names = [] if device_names is None else device_names
        with ProcessPoolExecutor() as executor:
            device_names.extend(
                executor.submit(query_devices, device_types).result()
            )

    device_types_set = list(set(get_device_types(device_names)))
    shark_module_paths_set = compile_stress_test_module(
        device_types_set, mlir_model, func_name, mlir_dialect
    )
    device_type_shark_module_path_map = {
        device_type: module_path
        for device_type, module_path in zip(
            device_types_set, shark_module_paths_set
        )
    }
    device_name_shark_module_path_map = {
        device_name: device_type_shark_module_path_map[
            get_device_type(device_name)
        ]
        for device_name in device_names
    }

    # This needs to run in a spearate process, because it uses the drvier chache
    # in IREE and a subsequent call to `iree.runtime.SystemContext.add_vm_module`
    # in a forked process will hang.
    with multiprocessing.Pool(
        len(device_name_shark_module_path_map) * oversubscription_factor
    ) as process_pool:
        process_pool.starmap(
            stress_test_compiled_model,
            [
                (
                    module_path,
                    func_name,
                    device_name,
                    inputs,
                    golden_out,
                    batch_size,
                    max_iterations,
                    max_duration_seconds,
                    inference_timeout_seconds,
                    tolerance_nulp,
                    stress_test_index,
                )
                for stress_test_index, (device_name, module_path) in enumerate(
                    list(device_name_shark_module_path_map.items())
                    * oversubscription_factor
                )
            ],
        )


if __name__ == "__main__":
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Downloads, compiles and runs a model from the tank to stress test the system."
    )
    parser.add_argument(
        "--model", type=str, help="Model name in the tank.", default="alexnet"
    )
    parser.add_argument(
        "--dynamic",
        help="Use dynamic version of the model.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--frontend", type=str, help="Frontend of the model.", default="torch"
    )
    parser.add_argument(
        "--mlir-dialect",
        type=str,
        help="MLIR dialect of the model.",
        default="linalg",
        choices=supported_dialects,
    )
    parser.add_argument(
        "--device-types",
        type=str,
        nargs="*",
        choices=_IREE_DEVICE_MAP.keys(),
        help="Runs the stress test on all devices with that type. "
        "If absent and no deveices are specified "
        "will run against all available devices.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="*",
        help="List of devices to run the stress test on. "
        "If device-types is specified will run against the union of the two.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of inputs to feed into the model",
        default=1,
    )
    parser.add_argument(
        "--oversubscription",
        type=int,
        help="Oversubscrption factor. Each device will execute the model simultaneously "
        "this many number of times.",
        default=1,
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum number of iterations to run the stress test per device.",
        default=10**7,
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Maximum number of seconds to run the stress test.",
        default=3600,
    )
    parser.add_argument(
        "--inference-timeout",
        type=float,
        help="Timeout in seconds for a single model inference operation.",
        default=60,
    )
    parser.add_argument(
        "--tolerance-nulp",
        type=int,
        help="The maximum number of unit in the last place for tolerance "
        "when verifing results with the golden reference output.",
        default=50000,
    )

    args = parser.parse_known_args()[0]
    stress_test(
        model_name=args.model,
        dynamic_model=args.dynamic,
        frontend=args.frontend,
        mlir_dialect=args.mlir_dialect,
        device_types=args.device_types,
        device_names=args.devices,
        batch_size=args.batch_size,
        oversubscription_factor=args.oversubscription,
        max_iterations=args.max_iterations,
        max_duration_seconds=args.max_duration,
        inference_timeout_seconds=args.inference_timeout,
        tolerance_nulp=args.tolerance_nulp,
    )
