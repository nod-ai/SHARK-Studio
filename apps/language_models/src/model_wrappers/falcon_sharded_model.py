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
    def __init__(self, decoder_layer_model, falcon_variant):
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
    def __init__(
        self, layer_id, device_idx, falcon_variant, device, precision
    ):
        super().__init__()
        self.layer_id = layer_id
        self.device_index = device_idx
        self.falcon_variant = falcon_variant
        self.device = device
        self.precision = precision

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
        import gc

        torch.cuda.empty_cache()
        gc.collect()
        from pathlib import Path
        from apps.language_models.utils import get_vmfb_from_path

        self.falcon_vmfb_path = Path(
            f"falcon_{self.falcon_variant}_layer_{self.layer_id}_{self.precision}_{self.device}.vmfb"
        )
        print("vmfb path for layer: ", self.falcon_vmfb_path)
        self.model = get_vmfb_from_path(
            self.falcon_vmfb_path,
            self.device,
            "linalg",
            device_id=self.device_index,
        )
        if self.model is None:
            raise ValueError("Layer vmfb not found")

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
            del self.model

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


class EightDecoderLayer(torch.nn.Module):
    def __init__(self, decoder_layer_model, falcon_variant):
        super().__init__()
        self.model = decoder_layer_model
        self.falcon_variant = falcon_variant

    def forward(self, hidden_states, attention_mask):
        new_pkvs = []
        for layer in self.model:
            outputs = layer(
                hidden_states=hidden_states,
                alibi=None,
                attention_mask=attention_mask,
                use_cache=True,
            )
            hidden_states = outputs[0]
            new_pkvs.append(
                (
                    outputs[-1][0],
                    outputs[-1][1],
                )
            )
        if self.falcon_variant == "7b":
            (
                (new_pkv00, new_pkv01),
                (new_pkv10, new_pkv11),
                (new_pkv20, new_pkv21),
                (new_pkv30, new_pkv31),
                (new_pkv40, new_pkv41),
                (new_pkv50, new_pkv51),
                (new_pkv60, new_pkv61),
                (new_pkv70, new_pkv71),
            ) = new_pkvs
            result = (
                hidden_states,
                new_pkv00,
                new_pkv01,
                new_pkv10,
                new_pkv11,
                new_pkv20,
                new_pkv21,
                new_pkv30,
                new_pkv31,
                new_pkv40,
                new_pkv41,
                new_pkv50,
                new_pkv51,
                new_pkv60,
                new_pkv61,
                new_pkv70,
                new_pkv71,
            )
        elif self.falcon_variant == "180b":
            (
                (new_pkv00, new_pkv01),
                (new_pkv10, new_pkv11),
                (new_pkv20, new_pkv21),
                (new_pkv30, new_pkv31),
                (new_pkv40, new_pkv41),
                (new_pkv50, new_pkv51),
                (new_pkv60, new_pkv61),
                (new_pkv70, new_pkv71),
                (new_pkv80, new_pkv81),
                (new_pkv90, new_pkv91),
                (new_pkv100, new_pkv101),
                (new_pkv110, new_pkv111),
                (new_pkv120, new_pkv121),
                (new_pkv130, new_pkv131),
                (new_pkv140, new_pkv141),
                (new_pkv150, new_pkv151),
                (new_pkv160, new_pkv161),
                (new_pkv170, new_pkv171),
                (new_pkv180, new_pkv181),
                (new_pkv190, new_pkv191),
            ) = new_pkvs
            result = (
                hidden_states,
                new_pkv00,
                new_pkv01,
                new_pkv10,
                new_pkv11,
                new_pkv20,
                new_pkv21,
                new_pkv30,
                new_pkv31,
                new_pkv40,
                new_pkv41,
                new_pkv50,
                new_pkv51,
                new_pkv60,
                new_pkv61,
                new_pkv70,
                new_pkv71,
                new_pkv80,
                new_pkv81,
                new_pkv90,
                new_pkv91,
                new_pkv100,
                new_pkv101,
                new_pkv110,
                new_pkv111,
                new_pkv120,
                new_pkv121,
                new_pkv130,
                new_pkv131,
                new_pkv140,
                new_pkv141,
                new_pkv150,
                new_pkv151,
                new_pkv160,
                new_pkv161,
                new_pkv170,
                new_pkv171,
                new_pkv180,
                new_pkv181,
                new_pkv190,
                new_pkv191,
            )
        else:
            raise ValueError(
                "Unsupported Falcon variant: ", self.falcon_variant
            )
        return result


class CompiledEightDecoderLayer(torch.nn.Module):
    def __init__(
        self, layer_id, device_idx, falcon_variant, device, precision
    ):
        super().__init__()
        self.layer_id = layer_id
        self.device_index = device_idx
        self.falcon_variant = falcon_variant
        self.device = device
        self.precision = precision

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
        import gc

        torch.cuda.empty_cache()
        gc.collect()
        from pathlib import Path
        from apps.language_models.utils import get_vmfb_from_path

        self.falcon_vmfb_path = Path(
            f"falcon_{self.falcon_variant}_layer_{self.layer_id}_{self.precision}_{self.device}.vmfb"
        )
        print("vmfb path for layer: ", self.falcon_vmfb_path)
        self.model = get_vmfb_from_path(
            self.falcon_vmfb_path,
            self.device,
            "linalg",
            device_id=self.device_index,
        )
        if self.model is None:
            raise ValueError("Layer vmfb not found")

        hidden_states = hidden_states.to(torch.float32).detach().numpy()
        attention_mask = attention_mask.detach().numpy()

        if alibi is not None or layer_past is not None:
            raise ValueError("Past Key Values and alibi should be None")
        else:
            output = self.model(
                "forward",
                (
                    hidden_states,
                    attention_mask,
                ),
            )
            del self.model

        if self.falcon_variant == "7b":
            result = (
                torch.tensor(output[0]),
                (
                    torch.tensor(output[1]),
                    torch.tensor(output[2]),
                ),
                (
                    torch.tensor(output[3]),
                    torch.tensor(output[4]),
                ),
                (
                    torch.tensor(output[5]),
                    torch.tensor(output[6]),
                ),
                (
                    torch.tensor(output[7]),
                    torch.tensor(output[8]),
                ),
                (
                    torch.tensor(output[9]),
                    torch.tensor(output[10]),
                ),
                (
                    torch.tensor(output[11]),
                    torch.tensor(output[12]),
                ),
                (
                    torch.tensor(output[13]),
                    torch.tensor(output[14]),
                ),
                (
                    torch.tensor(output[15]),
                    torch.tensor(output[16]),
                ),
            )
        elif self.falcon_variant == "180b":
            result = (
                torch.tensor(output[0]),
                (
                    torch.tensor(output[1]),
                    torch.tensor(output[2]),
                ),
                (
                    torch.tensor(output[3]),
                    torch.tensor(output[4]),
                ),
                (
                    torch.tensor(output[5]),
                    torch.tensor(output[6]),
                ),
                (
                    torch.tensor(output[7]),
                    torch.tensor(output[8]),
                ),
                (
                    torch.tensor(output[9]),
                    torch.tensor(output[10]),
                ),
                (
                    torch.tensor(output[11]),
                    torch.tensor(output[12]),
                ),
                (
                    torch.tensor(output[13]),
                    torch.tensor(output[14]),
                ),
                (
                    torch.tensor(output[15]),
                    torch.tensor(output[16]),
                ),
                (
                    torch.tensor(output[17]),
                    torch.tensor(output[18]),
                ),
                (
                    torch.tensor(output[19]),
                    torch.tensor(output[20]),
                ),
                (
                    torch.tensor(output[21]),
                    torch.tensor(output[22]),
                ),
                (
                    torch.tensor(output[23]),
                    torch.tensor(output[24]),
                ),
                (
                    torch.tensor(output[25]),
                    torch.tensor(output[26]),
                ),
                (
                    torch.tensor(output[27]),
                    torch.tensor(output[28]),
                ),
                (
                    torch.tensor(output[29]),
                    torch.tensor(output[30]),
                ),
                (
                    torch.tensor(output[31]),
                    torch.tensor(output[32]),
                ),
                (
                    torch.tensor(output[33]),
                    torch.tensor(output[34]),
                ),
                (
                    torch.tensor(output[35]),
                    torch.tensor(output[36]),
                ),
                (
                    torch.tensor(output[37]),
                    torch.tensor(output[38]),
                ),
                (
                    torch.tensor(output[39]),
                    torch.tensor(output[40]),
                ),
            )
        else:
            raise ValueError(
                "Unsupported Falcon variant: ", self.falcon_variant
            )
        return result


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
