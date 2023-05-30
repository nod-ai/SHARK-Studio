import torch


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


class CompiledFirstVicunaLayer(torch.nn.Module):
    def __init__(self, shark_module):
        super().__init__()
        self.model = shark_module

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
    ):
        hidden_states = hidden_states.detach()
        attention_mask = attention_mask.detach()
        position_ids = position_ids.detach()
        output = self.model(
            "forward",
            (
                hidden_states,
                attention_mask,
                position_ids,
            ),
        )

        output0 = torch.tensor(output[0])
        output1 = torch.tensor(output[1])
        output2 = torch.tensor(output[2])

        return (
            output0,
            (
                output1,
                output2,
            ),
        )


class CompiledSecondVicunaLayer(torch.nn.Module):
    def __init__(self, shark_module):
        super().__init__()
        self.model = shark_module

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions=False,
        use_cache=True,
    ):
        hidden_states = hidden_states.detach()
        attention_mask = attention_mask.detach()
        position_ids = position_ids.detach()
        pkv0 = past_key_value[0].detach()
        pkv1 = past_key_value[1].detach()
        output = self.model(
            "forward",
            (
                hidden_states,
                attention_mask,
                position_ids,
                pkv0,
                pkv1,
            ),
        )

        output0 = torch.tensor(output[0])
        output1 = torch.tensor(output[1])
        output2 = torch.tensor(output[2])

        return (
            output0,
            (
                output1,
                output2,
            ),
        )


class ShardedVicunaModel(torch.nn.Module):
    def __init__(self, model, layers0, layers1):
        super().__init__()
        self.model = model
        assert len(layers0) == len(model.model.layers)
        # self.model.model.layers = torch.nn.modules.container.ModuleList(layers0)
        self.model.model.config.use_cache = True
        self.model.model.config.output_attentions = False
        self.layers0 = layers0
        self.layers1 = layers1

    def forward(
        self,
        input_ids,
        is_first=True,
        past_key_values=None,
        attention_mask=None,
    ):
        if is_first:
            self.model.model.layers = torch.nn.modules.container.ModuleList(
                self.layers0
            )
            return self.model.forward(input_ids, attention_mask=attention_mask)
        else:
            self.model.model.layers = torch.nn.modules.container.ModuleList(
                self.layers1
            )
            return self.model.forward(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
