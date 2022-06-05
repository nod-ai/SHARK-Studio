import torch
from benchmarks.hf_transformer import SharkHFBenchmarkRunner

if __name__ == "__main__":
    model_name = "microsoft/MiniLM-L12-H384-uncased"
    test_input = torch.randint(2, (1, 128))
    shark_module = SharkHFBenchmarkRunner(model_name, (test_input,), device="cpu",
                                  jit_trace=True)
    shark_module.benchmark_c()
    shark_module.benchmark_python((test_input,))
    shark_module.benchmark_torch(test_input)
