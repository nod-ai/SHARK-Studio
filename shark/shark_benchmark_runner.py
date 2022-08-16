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

from shark.shark_runner import SharkRunner
from shark.iree_utils.compile_utils import export_iree_module_to_vmfb
from shark.iree_utils.benchmark_utils import (
    build_benchmark_args,
    run_benchmark_module,
)
from shark.parser import shark_args
from datetime import datetime
import time
import csv
import os


class SharkBenchmarkRunner(SharkRunner):
    # SharkRunner derived class with Benchmarking capabilities.
    def __init__(
        self,
        mlir_module: str,
        function_name: str = "forward",
        device: str = "none",
        mlir_dialect: str = "linalg",
    ):
        self.device = shark_args.device if device == "none" else device
        self.frontend_model = None
        self.vmfb_file = None
        self.mlir_dialect = mlir_dialect
        SharkRunner.__init__(
            self,
            mlir_module,
            function_name,
            device,
            self.mlir_dialect,
        )
        if self.vmfb_file == None:
            self.vmfb_file = export_iree_module_to_vmfb(
                mlir_module, device, shark_args.repro_dir, self.mlir_dialect
            )

    def setup_cl(self, input_tensors):
        self.benchmark_cl = build_benchmark_args(
            self.vmfb_file,
            self.device,
            input_tensors,
            mlir_dialect=self.mlir_dialect,
        )
        # print(self.benchmark_cl)

    def benchmark_frontend(self, modelname):
        if self.mlir_dialect in ["linalg", "torch"]:
            return self.benchmark_torch(modelname)
        elif self.mlir_dialect in ["mhlo", "tf"]:
            return self.benchmark_tf(modelname)

    def benchmark_torch(self, modelname):
        import torch
        from tank.model_utils import get_torch_model

        if self.device == "gpu":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        torch_device = torch.device(
            "cuda:0" if self.device == "gpu" else "cpu"
        )
        HFmodel, input = get_torch_model(modelname)[:2]
        frontend_model = HFmodel.model
        frontend_model.to(torch_device)
        input.to(torch_device)

        for i in range(shark_args.num_warmup_iterations):
            frontend_model.forward(input)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = frontend_model.forward(input)
            if i == shark_args.num_iterations - 1:
                end = time.time()
                break
        print(
            f"Torch benchmark:{shark_args.num_iterations/(end-begin)} iter/second, Total Iterations:{shark_args.num_iterations}"
        )
        return [
            f"{shark_args.num_iterations/(end-begin)}",
            f"{((end-begin)/shark_args.num_iterations)*1000}",
        ]

    def benchmark_tf(self, modelname):
        import tensorflow as tf
        from tank.model_utils_tf import get_tf_model

        model, input, = get_tf_model(
            modelname
        )[:2]
        frontend_model = model

        for i in range(shark_args.num_warmup_iterations):
            frontend_model.forward(*input)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = frontend_model.forward(*input)
            if i == shark_args.num_iterations - 1:
                end = time.time()
                break
        print(
            f"TF benchmark:{shark_args.num_iterations/(end-begin)} iter/second, Total Iterations:{shark_args.num_iterations}"
        )
        return [
            f"{shark_args.num_iterations/(end-begin)}",
            f"{((end-begin)/shark_args.num_iterations)*1000}",
        ]

    def benchmark_c(self):
        print(self.benchmark_cl)
        result = run_benchmark_module(self.benchmark_cl)
        print(f"Shark-IREE-C benchmark:{result} iter/second")
        return [f"{result}", f"{1000/result}"]

    def benchmark_python(self, inputs):
        input_list = [x for x in inputs]
        for i in range(shark_args.num_warmup_iterations):
            self.run(input_list)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = self.run(input_list)
            if i == shark_args.num_iterations - 1:
                end = time.time()
        print(
            f"Shark-IREE Python benchmark:{shark_args.num_iterations/(end-begin)} iter/second, Total Iterations:{shark_args.num_iterations}"
        )
        return [
            f"{shark_args.num_iterations/(end-begin)}",
            f"{((end-begin)/shark_args.num_iterations)*1000}",
        ]

    def benchmark_all_csv(
        self, inputs: tuple, modelname, dynamic, device_str, frontend
    ):
        self.setup_cl(inputs)
        field_names = [
            "model",
            "engine",
            "dynamic",
            "dialect",
            "device",
            "iter/sec",
            "ms/iter",
            "iterations",
            "datetime",
        ]
        engines = ["frontend", "shark_python", "shark_iree_c"]

        if not os.path.exists("bench_results.csv"):
            with open("bench_results.csv", mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(field_names)

        with open("bench_results.csv", mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            bench_result = {}
            bench_result["model"] = modelname
            if dynamic == True:
                bench_result["dynamic"] = "True"
            else:
                bench_result["dynamic"] = "False"
            bench_result["device"] = device_str
            for e in engines:
                if e == "frontend":
                    bench_result["engine"] = frontend
                    bench_result["iter/sec"] = self.benchmark_frontend(
                        modelname
                    )[0]
                    bench_result["ms/iter"] = self.benchmark_frontend(
                        modelname
                    )[1]
                elif e == "shark_python":
                    bench_result["engine"] = "shark_python"
                    bench_result["iter/sec"] = self.benchmark_python(inputs)[0]
                    bench_result["ms/iter"] = self.benchmark_python(inputs)[1]
                else:
                    bench_result["engine"] = "shark_iree_c"
                    bench_result["iter/sec"] = self.benchmark_c()[0]
                    bench_result["ms/iter"] = self.benchmark_c()[1]
                bench_result["dialect"] = self.mlir_dialect
                bench_result["iterations"] = shark_args.num_iterations
                bench_result["datetime"] = str(datetime.now())
                writer.writerow(bench_result)
