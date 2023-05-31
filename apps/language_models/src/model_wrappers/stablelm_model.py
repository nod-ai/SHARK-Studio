import torch


class StableLMModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        combine_input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        output = self.model(**combine_input_dict)
        return output.logits
