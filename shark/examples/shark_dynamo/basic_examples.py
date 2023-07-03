import torch
import shark


def foo(x, a):
    if x.shape[0] > 3:
        return x + a
    else:
        return x + 3


shark_options = {"device": "cpu"}
compiled = torch.compile(foo, backend="shark", options=shark_options)

input = torch.ones(4)

x = compiled(input, input)

print(x)

input = torch.ones(3)

x = compiled(input, input)

print(x)
