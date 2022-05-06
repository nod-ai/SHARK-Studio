import torch
from functorch.compile import aot_function, nop
from functorch import make_fx
from torch.nn.utils import _stateless
from shark.shark_runner import SharkTrainer

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.fc(x)

mod = Foo()

def get_sorted_params(named_params):
    return [i[1] for i in sorted(named_params.items())]

inp = (torch.randn(3, 3),)

def forward(params, buffers, args):
    params_and_buffers = {**params, **buffers}
    _stateless.functional_call(mod, params_and_buffers, args, {}).sum().backward()
    optim = torch.optim.SGD(get_sorted_params(params), lr=0.01)
    optim.step()
    return params, buffers


fx_graph = forward(dict(mod.named_parameters()),
                            dict(mod.named_buffers()), inp)

shark_module = SharkTrainer(mod, inp, custom_inference_fn=forward)

print(shark_module.forward())
