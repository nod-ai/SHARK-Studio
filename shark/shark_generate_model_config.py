import json
from collections import OrderedDict


class GenerateConfigFile:
    def __init__(
        self,
        model,
        num_sharding_stages: int,
        sharding_stages_id: list[str] = None,
    ):
        self.model = model
        self.num_sharding_stages = num_sharding_stages
        self.sharding_stages_id = sharding_stages_id
        assert self.num_sharding_stages == len(
            self.sharding_stages_id
        ), "Number of sharding stages should be equal to the list of their ID"

    def generate_json(self):
        model_dictionary = dict()

        for name, m in self.model.named_modules():
            if name == "":
                continue

            # Remove non-leaf nodes from the config as they aren't an operation
            substring_before_final_period = name.split(".")[:-1]
            substring_before_final_period = ".".join(
                substring_before_final_period
            )
            if substring_before_final_period in model_dictionary:
                del model_dictionary[substring_before_final_period]

            layer_dict = {n: "None" for n in self.sharding_stages_id}
            model_dictionary[name] = layer_dict

        with open("model_config.json", "w") as outfile:
            json.dump(model_dictionary, outfile)
