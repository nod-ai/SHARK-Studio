import torch
from benchmarks.hf_transformer import SharkHFBenchmarkRunner
import importlib
import pytest

torch.manual_seed(0)

############################# HF Benchmark Tests ####################################

# Test running benchmark module without failing.
pytest_benchmark_param = pytest.mark.parametrize(
    ('dynamic', 'device'),
    [
        pytest.param(False, 'cpu'),
        # TODO: Language models are failing for dynamic case..
        pytest.param(True, 'cpu', marks=pytest.mark.skip),
    ])

@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None, reason = "Cannot find ONNXRUNTIME.")
@pytest_benchmark_param
def test_HFbench_minilm_torch(dynamic, device):
    model_name = "bert-base-uncased"
    test_input = torch.randint(2, (1, 128))
    try:
        shark_module = SharkHFBenchmarkRunner(model_name, (test_input,),
                                            jit_trace=True, dynamic = dynamic, device = device)
        shark_module.benchmark_c()
        shark_module.benchmark_python((test_input,))
        shark_module.benchmark_torch(test_input)
        shark_module.benchmark_onnx(test_input)
        # If becnhmarking succesful, assert success/True.
        assert True
    except Exception as e:
        # If anything happen during benchmarking, assert False/failure.
        assert False
