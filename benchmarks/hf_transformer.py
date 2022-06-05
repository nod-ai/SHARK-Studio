import torch
from shark.shark_runner import SharkBenchmarkRunner
from shark.parser import shark_args
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime.transformers.benchmark as onnx_benchmark
from onnxruntime.transformers.benchmark_helper import ConfigModifier, Precision
import os
import psutil

class HuggingFaceLanguage(torch.nn.Module):

    def __init__(self, hf_model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            num_labels=
            2,  # The number of output labels--2 for binary classification.
            output_attentions=
            False,  # Whether the model returns attentions weights.
            output_hidden_states=
            False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]

class SharkHFBenchmarkRunner(SharkBenchmarkRunner):
    # SharkRunner derived class with Benchmarking capabilities.
    def __init__(
        self,
        model_name: str,
        input: tuple,
        dynamic: bool = False,
        device: str = "cpu",
        jit_trace: bool = False,
        from_aot: bool = False,
        frontend: str = "torch",
    ):
        self.device = device
        self.model_name = model_name
        model = HuggingFaceLanguage(model_name)
        SharkBenchmarkRunner.__init__(self, model, input, dynamic, device, jit_trace,
                             from_aot, frontend)

    def benchmark_torch(self, inputs):
        use_gpu = self.device == "gpu"
        # Set set the model's layer number to automatic.
        config_modifier = ConfigModifier(None)
        num_threads = psutil.cpu_count(logical=False)
        batch_sizes = [inputs.shape[0]]
        sequence_lengths = [inputs.shape[-1]]
        cache_dir = os.path.join(".", "cache_models")
        verbose = False
        result = onnx_benchmark.run_pytorch(
            use_gpu,
            [self.model_name],
            None,
            config_modifier,
            Precision.FLOAT32,
            num_threads,
            batch_sizes,
            sequence_lengths,
            shark_args.num_iterations,
            False,
            cache_dir,
            verbose,
        )
        print(
            f"ONNX Pytorch-benchmark:{result[0]['QPS']} iter/second, Total Iterations:{shark_args.num_iterations}"
        )
