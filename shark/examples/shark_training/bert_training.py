import torch
from torch.nn.utils import _stateless
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shark.shark_runner import SharkTrainer


class MiniLMSequenceClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased",  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


mod = MiniLMSequenceClassification()


def get_sorted_params(named_params):
    return [i[1] for i in sorted(named_params.items())]


print(dict(mod.named_buffers()))

inp = (torch.randint(2, (1, 128)),)


def forward(params, buffers, args):
    params_and_buffers = {**params, **buffers}
    _stateless.functional_call(
        mod, params_and_buffers, args, {}
    ).sum().backward()
    optim = torch.optim.SGD(get_sorted_params(params), lr=0.01)
    # optim.load_state_dict(optim_state)
    optim.step()
    return params, buffers


shark_module = SharkTrainer(mod, inp, custom_inference_fn=forward)

print(shark_module.forward())
