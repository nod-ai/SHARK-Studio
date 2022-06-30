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
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = None,
        jit_trace: bool = False,
        from_aot: bool = False,
        frontend: str = "torch",
    ):
        SharkRunner.__init__(self, model, input, dynamic, device, jit_trace, from_aot, frontend)
        if self.vmfb_file == None:
            self.vmfb_file = export_iree_module_to_vmfb(self.model, device, shark_args.repro_dir, frontend)
        self.benchmark_cl = build_benchmark_args(self.vmfb_file, device, input, frontend, from_aot)

    def benchmark_frontend(self, inputs):
        if self.frontend in ["pytorch", "torch"]:
            return self.benchmark_torch(inputs)
        elif self.frontend in ["tensorflow", "tf"]:
            return self.benchmark_tf(inputs)

    def benchmark_torch(self, inputs):
        inputs = self.input if self.from_aot else inputs
        inputs = inputs[0]
        for i in range(shark_args.num_warmup_iterations):
            self.frontend_model.forward(inputs)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = self.frontend_model.forward(inputs)
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

    def benchmark_tf(self, inputs):
        for i in range(shark_args.num_warmup_iterations):
            self.frontend_model.forward(*inputs)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = self.frontend_model.forward(*inputs)
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
        result = run_benchmark_module(self.benchmark_cl)
        print(f"Shark-{self.frontend} C-benchmark:{result} iter/second")
        return [f"{result}", f"{1000/result}"]

    def benchmark_python(self, inputs):
        inputs = self.input if self.from_aot else inputs
        input_list = [x for x in inputs]
        for i in range(shark_args.num_warmup_iterations):
            self.forward(input_list, self.frontend)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = self.forward(input_list, self.frontend)
            if i == shark_args.num_iterations - 1:
                end = time.time()
        print(
            f"Shark-{self.frontend} Python-benchmark:{shark_args.num_iterations/(end-begin)} iter/second, Total Iterations:{shark_args.num_iterations}"
        )
        return [
            f"{shark_args.num_iterations/(end-begin)}",
            f"{((end-begin)/shark_args.num_iterations)*1000}",
        ]

    def benchmark_all(self, inputs):
        self.benchmark_frontend(inputs)
        self.benchmark_python(inputs)
        self.benchmark_c()

    def benchmark_all_csv(self, inputs, modelname, dynamic, device_str):
        field_names = [
            "platform",
            "model",
            "dynamic",
            "device",
            "iter/sec",
            "ms/iter",
            "datetime",
        ]
        platforms = ["frontend", "shark_python", "shark_iree_c"]

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
            for p in platforms:
                if p == "frontend":
                    bench_result["platform"] = "frontend"
                    bench_result["iter/sec"] = self.benchmark_frontend(inputs)[0]
                    bench_result["ms/iter"] = self.benchmark_frontend(inputs)[1]
                elif p == "shark_python":
                    bench_result["platform"] = "shark_python"
                    bench_result["iter/sec"] = self.benchmark_python(inputs)[0]
                    bench_result["ms/iter"] = self.benchmark_python(inputs)[1]
                else:
                    bench_result["platform"] = "shark_iree_c"
                    bench_result["iter/sec"] = self.benchmark_c()[0]
                    bench_result["ms/iter"] = self.benchmark_c()[1]
                bench_result["datetime"] = str(datetime.now())
                writer.writerow(bench_result)
