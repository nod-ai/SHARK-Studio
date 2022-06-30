import torch
from shark.parser import parser
from benchmarks.hf_transformer import SharkHFBenchmarkRunner

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
    shark_module = SharkHFBenchmarkRunner(model_name, (test_input,), jit_trace=True)
    shark_module.benchmark_c()
    shark_module.benchmark_python((test_input,))
    shark_module.benchmark_torch(test_input)
    shark_module.benchmark_onnx(test_input)
