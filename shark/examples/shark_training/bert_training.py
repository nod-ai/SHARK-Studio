import torch
import time
import numpy as np
from torch.nn.utils import _stateless
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertModel,
    AutoModelForMaskedLM,
)
from shark.shark_trainer import SharkTrainer


def get_torch_params(model):
    params = {v: i for v, i in model.named_parameters()}
    buffers = {v: i for v, i in model.named_buffers()}
    return params, buffers


class MiniLMSequenceClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            "bert-large-uncased",  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


def get_sorted_params(named_params):
    return [i[1] for i in sorted(named_params.items())]


def get_model_and_test_values():
    mod = MiniLMSequenceClassification()  # .to("cuda")
    inp = torch.randint(2, (32, 128))  # .to("cuda")

    training_inputs = [i.detach() for i in mod.parameters()]
    for i in mod.buffers():
        training_inputs.append(i.detach())

    training_inputs.append(inp.detach())
    # np.savez("/home/dan/inputs.npz", *[x.numpy() for x in training_inputs])

    def forward(params, buffers, args):
        params_and_buffers = {**params, **buffers}
        _stateless.functional_call(
            mod, params_and_buffers, args, {}
        ).sum().backward()
        optim = torch.optim.SGD(get_sorted_params(params), lr=0.01)
        # optim.load_state_dict(optim_state)
        optim.step()
        return params, buffers

    shark_module = SharkTrainer(mod, (inp,), from_aot=True)
    shark_module.compile(forward)

    rr = shark_module.shark_runner
    asm = rr.mlir_module.operation.get_asm()
    return asm, training_inputs, forward, "forward"


def custom_benchmark_func(mod, shark_module):
    p, b = get_torch_params(mod)
    shark_params_and_buffers = shark_module.shark_runner.run(training_inputs)
    iterations = 1
    start = time.time()
    for i in range(iterations):
        p, b = forward(p, b, inp)
    end = time.time()
    total_time = end - start
    print("total_time(ms)/iter: " + str(1000 * total_time / iterations))
    golden = [v for v in p.values()][0].shape
    test = shark_params_and_buffers[0].shape
    return np.allclose(golden, test)
