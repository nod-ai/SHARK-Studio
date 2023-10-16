import torch
from typing import Optional, Tuple


class WordEmbeddingsLayer(torch.nn.Module):
    def __init__(self, word_embedding_layer):
        super().__init__()
        self.model = word_embedding_layer

    def forward(self, input_ids):
        output = self.model.forward(input=input_ids)
        return output


class CompiledWordEmbeddingsLayer(torch.nn.Module):
    def __init__(self, compiled_word_embedding_layer):
        super().__init__()
        self.model = compiled_word_embedding_layer

    def forward(self, input_ids):
        input_ids = input_ids.detach().numpy()
        new_input_ids = self.model("forward", input_ids)
        new_input_ids = new_input_ids.reshape(
            [1, new_input_ids.shape[0], new_input_ids.shape[1]]
        )
        return torch.tensor(new_input_ids)


class LNFEmbeddingLayer(torch.nn.Module):
    def __init__(self, ln_f):
        super().__init__()
        self.model = ln_f

    def forward(self, hidden_states):
        output = self.model.forward(input=hidden_states)
        return output


class CompiledLNFEmbeddingLayer(torch.nn.Module):
    def __init__(self, ln_f):
        super().__init__()
        self.model = ln_f

    def forward(self, hidden_states):
        hidden_states = hidden_states.detach().numpy()
        new_hidden_states = self.model("forward", (hidden_states,))

        return torch.tensor(new_hidden_states)


class LMHeadEmbeddingLayer(torch.nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.model = embedding_layer

    def forward(self, hidden_states):
        output = self.model.forward(input=hidden_states)
        return output


class CompiledLMHeadEmbeddingLayer(torch.nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.model = lm_head

    def forward(self, hidden_states):
        hidden_states = hidden_states.detach().numpy()
        new_hidden_states = self.model("forward", (hidden_states,))
        return torch.tensor(new_hidden_states)


class DecoderLayer(torch.nn.Module):
    def __init__(self, decoder_layer_model):
        super().__init__()
        self.model = decoder_layer_model

    def forward(self, hidden_states, attention_mask):
        output = self.model.forward(
            hidden_states=hidden_states,
            alibi=None,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return (output[0], output[1][0], output[1][1])


class CompiledDecoderLayer(torch.nn.Module):
    def __init__(self, shark_decoder_layer_module):
        super().__init__()
        self.model = shark_decoder_layer_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        alibi: torch.Tensor = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        hidden_states = hidden_states.to(torch.float32).detach().numpy()
        attention_mask = attention_mask.detach().numpy()

        if alibi is not None or layer_past is not None:
            raise ValueError("Past Key Values and alibi should be None")
        else:
            new_hidden_states, pkv1, pkv2 = self.model(
                "forward",
                (
                    hidden_states,
                    attention_mask,
                ),
            )
            return tuple(
                [
                    torch.tensor(new_hidden_states),
                    tuple(
                        [
                            torch.tensor(pkv1),
                            torch.tensor(pkv2),
                        ]
                    ),
                ]
            )


class ShardedFalconModel:
    def __init__(self, model, layers, word_embeddings, ln_f, lm_head):
        super().__init__()
        self.model = model
        self.model.transformer.h = torch.nn.modules.container.ModuleList(
            layers
        )
        self.model.transformer.word_embeddings = word_embeddings
        self.model.transformer.ln_f = ln_f
        self.model.lm_head = lm_head

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits[:, -1, :]
