import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shark.shark_inference import SharkInference

torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")


class MiniLMSequenceClassification(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased",  # The pretrained model.
            num_labels=
            2,  # The number of output labels--2 for binary classification.
            output_attentions=
            False,  # Whether the model returns attentions weights.
            output_hidden_states=
            False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


test_input = torch.randint(2, (1, 128))

shark_module = SharkInference(MiniLMSequenceClassification(), (test_input,),
                              jit_trace=True, benchmark_mode=True)

shark_module.compile()
shark_module.forward((test_input,))
shark_module.benchmark_all((test_input,))
