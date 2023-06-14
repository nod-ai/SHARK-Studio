import torch


class FalconModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "use_cache": True,
        }
        output = self.model(
            **input_dict,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )[0]
        return output[:, -1, :]
