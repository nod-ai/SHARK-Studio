import torch
import torch_mlir
import torch._dynamo as torchdynamo
from shark.sharkdynamo.utils import make_shark_compiler


import warnings, logging

warnings.simplefilter("ignore")
torchdynamo.config.log_level = logging.ERROR


torchdynamo.reset()


@torchdynamo.optimize(
    make_shark_compiler(use_tracing=False, device="cuda", verbose=False)
)
def foo(t):
    return 2 * t


example_input = torch.rand((2, 3))
x = foo(example_input)
print(x)


torchdynamo.reset()


@torchdynamo.optimize(
    make_shark_compiler(use_tracing=False, device="cuda", verbose=False)
)
def foo(a, b):
    x = a / (a + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


print(foo(torch.rand((2, 3)), -torch.rand((2, 3))))


torchdynamo.reset()


@torchdynamo.optimize(
    make_shark_compiler(use_tracing=False, device="cuda", verbose=True)
)
def foo(a):
    for i in range(10):
        a += 1.0
    return a


print(foo(torch.rand((1, 2))))

torchdynamo.reset()


@torchdynamo.optimize(
    make_shark_compiler(use_tracing=False, device="cuda", verbose=True)
)
def test_unsupported_types(t, y):
    return t, 2 * y


str_input = "hello"
tensor_input = torch.randn(2)
print(test_unsupported_types(str_input, tensor_input))
