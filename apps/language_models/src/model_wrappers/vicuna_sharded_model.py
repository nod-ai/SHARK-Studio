import torch
import time


class FirstVicunaLayer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, attention_mask, position_ids):
        outputs = self.model(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        next_hidden_states = outputs[0]
        past_key_value_out0, past_key_value_out1 = (
            outputs[-1][0],
            outputs[-1][1],
        )

        return (
            next_hidden_states,
            past_key_value_out0,
            past_key_value_out1,
        )


class SecondVicunaLayer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value0,
        past_key_value1,
    ):
        outputs = self.model(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(
                past_key_value0,
                past_key_value1,
            ),
            use_cache=True,
        )
        next_hidden_states = outputs[0]
        past_key_value_out0, past_key_value_out1 = (
            outputs[-1][0],
            outputs[-1][1],
        )

        return (
            next_hidden_states,
            past_key_value_out0,
            past_key_value_out1,
        )


class ShardedVicunaModel(torch.nn.Module):
    def __init__(self, model, layers, lmhead, embedding, norm):
        super().__init__()
        self.model = model
        self.model.model.config.use_cache = True
        self.model.model.config.output_attentions = False
        self.layers = layers
        self.norm = norm
        self.embedding = embedding
        self.lmhead = lmhead
        self.model.model.norm = self.norm
        self.model.model.embed_tokens = self.embedding
        self.model.lm_head = self.lmhead
        self.model.model.layers = torch.nn.modules.container.ModuleList(
            self.layers
        )

    def forward(
        self,
        input_ids,
        is_first=True,
        past_key_values=None,
        attention_mask=None,
    ):
        return self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )


class LMHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states):
        output = self.model(hidden_states)
        return output


class LMHeadCompiled(torch.nn.Module):
    def __init__(self, shark_module):
        super().__init__()
        self.model = shark_module

    def forward(self, hidden_states):
        hidden_states_sample = hidden_states.detach()

        output = self.model("forward", (hidden_states,))
        output = torch.tensor(output)

        return output


class VicunaNorm(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states):
        output = self.model(hidden_states)
        return output


class VicunaNormCompiled(torch.nn.Module):
    def __init__(self, shark_module):
        super().__init__()
        self.model = shark_module

    def forward(self, hidden_states):
        try:
            hidden_states.detach()
        except:
            pass
        output = self.model("forward", (hidden_states,), send_to_host=True)
        output = torch.tensor(output)

        return output


class VicunaEmbedding(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        output = self.model(input_ids)
        return output


class VicunaEmbeddingCompiled(torch.nn.Module):
    def __init__(self, shark_module):
        super().__init__()
        self.model = shark_module

    def forward(self, input_ids):
        input_ids.detach()
        output = self.model("forward", (input_ids,), send_to_host=True)
        output = torch.tensor(output)

        return output


class CompiledVicunaLayer(torch.nn.Module):
    def __init__(self, shark_module, idx, breakpoints):
        super().__init__()
        self.model = shark_module
        self.idx = idx
        self.breakpoints = breakpoints

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
    ):
        if self.breakpoints is None:
            is_breakpoint = False
        else:
            is_breakpoint = self.idx + 1 in self.breakpoints
        if past_key_value is None:
            output = self.model(
                "first_vicuna_forward",
                (
                    hidden_states,
                    attention_mask,
                    position_ids,
                ),
                send_to_host=is_breakpoint,
            )

            if is_breakpoint:
                output0 = torch.tensor(output[0])
                output1 = torch.tensor(output[1])
                output2 = torch.tensor(output[2])
            else:
                output0 = output[0]
                output1 = output[1]
                output2 = output[2]

            return (
                output0,
                (
                    output1,
                    output2,
                ),
            )
        else:
            pkv0 = past_key_value[0]
            pkv1 = past_key_value[1]
            output = self.model(
                "second_vicuna_forward",
                (
                    hidden_states,
                    attention_mask,
                    position_ids,
                    pkv0,
                    pkv1,
                ),
                send_to_host=is_breakpoint,
            )

            if is_breakpoint:
                output0 = torch.tensor(output[0])
                output1 = torch.tensor(output[1])
                output2 = torch.tensor(output[2])
            else:
                output0 = output[0]
                output1 = output[1]
                output2 = output[2]

            return (
                output0,
                (
                    output1,
                    output2,
                ),
            )
