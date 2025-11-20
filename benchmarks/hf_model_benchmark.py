import torch
from amdshark.parser import parser
from benchmarks.hf_transformer import AMDSharkHFBenchmarkRunner

parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help='Specifies name of HF model to benchmark. (For exmaple "microsoft/MiniLM-L12-H384-uncased"',
)
load_args, unknown = parser.parse_known_args()

if __name__ == "__main__":
    model_name = load_args.model_name
    test_input = torch.randint(2, (1, 128))
    amdshark_module = AMDSharkHFBenchmarkRunner(
        model_name, (test_input,), jit_trace=True
    )
    amdshark_module.benchmark_c()
    amdshark_module.benchmark_python((test_input,))
    amdshark_module.benchmark_torch(test_input)
    amdshark_module.benchmark_onnx(test_input)
