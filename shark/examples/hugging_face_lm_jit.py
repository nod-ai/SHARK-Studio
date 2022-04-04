import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shark_runner import SharkInference

torch.manual_seed(0)

# "distilbert-base-uncased"
models = ["albert-base-v2", "bert-base-uncased"]


def prepare_sentence_tokens(tokenizer, sentence):
    return torch.tensor([tokenizer.encode(sentence)])


class HuggingFaceLanguage(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


for hf_model in models:
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    test_input = prepare_sentence_tokens(
        tokenizer, "this project is very interesting"
    )
    shark_module = SharkInference(
        HuggingFaceLanguage(hf_model),
        (test_input,),
        jit_trace=True,
    )
    shark_module.forward((test_input,))
