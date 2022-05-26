import torch
from torch.nn.utils import _stateless
from shark.shark_runner import SharkTrainer


class Foo(torch.nn.Module):

    def __init__(self):
        super(Foo, self).__init__()
        self.l1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(16, 2)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


mod = Foo()


def get_sorted_params(named_params):
    return [i[1] for i in sorted(named_params.items())]


inp = (torch.randn(10, 10),)


def forward(params, buffers, args):
    params_and_buffers = {**params, **buffers}
    _stateless.functional_call(mod, params_and_buffers, args,
                               {}).sum().backward()
    optim = torch.optim.SGD(get_sorted_params(params), lr=0.01)
    optim.step()
    return params, buffers


fx_graph = forward(dict(mod.named_parameters()), dict(mod.named_buffers()), inp)

shark_module = SharkTrainer(mod, inp, custom_inference_fn=forward)

shark_module.train(num_iters=10)
