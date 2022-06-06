import torch
from shark.shark_runner import SharkBenchmarkRunner
from shark.parser import shark_args
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.transformers.benchmark import run_pytorch, run_tensorflow, run_onnxruntime
from onnxruntime.transformers.huggingface_models import MODELS
from onnxruntime.transformers.benchmark_helper import ConfigModifier, Precision
import os
import psutil


class OnnxFusionOptions(object):

    def __init__(self):
        self.disable_gelu = False
        self.disable_layer_norm = False
        self.disable_attention = False
        self.disable_skip_layer_norm = False
        self.disable_embed_layer_norm = False
        self.disable_bias_skip_layer_norm = False
        self.disable_bias_gelu = False
        self.enable_gelu_approximation = False
        self.use_mask_index = False
        self.no_attention_mask = False


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
        device: str = None,
        jit_trace: bool = False,
        from_aot: bool = False,
        frontend: str = "torch",
    ):
        self.device = device if device is not None else shark_args.device
        if self.device == "gpu":
            raise ValueError(
                "Currently GPU Benchmarking is not supported due to OOM from ORT."
            )
        self.model_name = model_name
        model = HuggingFaceLanguage(model_name)
        SharkBenchmarkRunner.__init__(self, model, input, dynamic, self.device,
                                      jit_trace, from_aot, frontend)

    def benchmark_torch(self, inputs):
        use_gpu = self.device == "gpu"
        # Set set the model's layer number to automatic.
        config_modifier = ConfigModifier(None)
        num_threads = psutil.cpu_count(logical=False)
        batch_sizes = [inputs.shape[0]]
        sequence_lengths = [inputs.shape[-1]]
        cache_dir = os.path.join(".", "cache_models")
        verbose = False
        result = run_pytorch(use_gpu, [self.model_name], None, config_modifier,
                             Precision.FLOAT32, num_threads, batch_sizes,
                             sequence_lengths, shark_args.num_iterations, False,
                             cache_dir, verbose)
        print(
            f"ONNX Pytorch-benchmark:{result[0]['QPS']} iter/second, Total Iterations:{shark_args.num_iterations}"
        )

    # TODO: Currently non-functional due to TF runtime error. There might be some issue with, initializing TF.
    def benchmark_tf(self, inputs):
        use_gpu = self.device == "gpu"
        # Set set the model's layer number to automatic.
        config_modifier = ConfigModifier(None)
        num_threads = psutil.cpu_count(logical=False)
        batch_sizes = [inputs.shape[0]]
        sequence_lengths = [inputs.shape[-1]]
        cache_dir = os.path.join(".", "cache_models")
        verbose = False
        result = run_tensorflow(use_gpu, [self.model_name], None,
                                config_modifier, Precision.FLOAT32, num_threads,
                                batch_sizes, sequence_lengths,
                                shark_args.num_iterations, cache_dir, verbose)
        print(
            f"ONNX TF-benchmark:{result[0]['QPS']} iter/second, Total Iterations:{shark_args.num_iterations}"
        )

    def benchmark_onnx(self, inputs):
        if self.model_name not in MODELS:
            print(
                f"{self.model_name} is currently not supported in ORT's HF. Check \
https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/huggingface_models.py \
for currently supported models. Exiting benchmark ONNX.")
            return
        use_gpu = self.device == "gpu"
        num_threads = psutil.cpu_count(logical=False)
        batch_sizes = [inputs.shape[0]]
        sequence_lengths = [inputs.shape[-1]]
        cache_dir = os.path.join(".", "cache_models")
        onnx_dir = os.path.join(".", "onnx_models")
        verbose = False
        input_counts = [1]
        optimize_onnx = True
        validate_onnx = False
        disable_ort_io_binding = False
        use_raw_attention_mask = True
        model_fusion_statistics = {}
        overwrite = False
        model_source = "pt"  #Either "pt" or "tf"
        provider = None
        config_modifier = ConfigModifier(None)
        onnx_args = OnnxFusionOptions()
        result = run_onnxruntime(
            use_gpu, provider, [self.model_name], None, config_modifier,
            Precision.FLOAT32, num_threads, batch_sizes, sequence_lengths,
            shark_args.num_iterations, input_counts, optimize_onnx,
            validate_onnx, cache_dir, onnx_dir, verbose, overwrite,
            disable_ort_io_binding, use_raw_attention_mask,
            model_fusion_statistics, model_source, onnx_args)
        print(
            f"ONNX ORT-benchmark:{result[0]['QPS']} iter/second, Total Iterations:{shark_args.num_iterations}"
        )
