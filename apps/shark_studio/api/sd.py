from turbine_models.custom_models.sd_inference import clip, unet, vae
from shark.iree_utils.compile_utils import get_iree_compiled_module
from apps.shark_studio.api.utils import get_resource_path
import iree.runtime as ireert
import gc
import torch

sd_model_map = {
    "sd15": {
        "base_model_id": "runwayml/stable-diffusion-v1-5"
        "clip": {
            "initializer": clip.export_clip_model,
            "max_tokens": 77,
        }
        "unet": {
            "initializer": unet.export_unet_model,
            "max_tokens": 512,
        }
        "vae_decode": {
            "initializer": vae.export_vae_model,,
        }
    }
}


class SharkStableDiffusionPipeline:
    def __init__(
        self, model_name, , device=None, precision="fp32"
    ):
        print(sd_model_map[model_name])
        self.hf_model_name = llm_model_map[model_name]["hf_model_name"]
        self.torch_ir, self.tokenizer = llm_model_map[model_name][
            "initializer"
        ](self.hf_model_name, hf_auth_token, compile_to="torch")
        self.tempfile_name = get_resource_path("llm.torch.tempfile")
        with open(self.tempfile_name, "w+") as f:
            f.write(self.torch_ir)
        del self.torch_ir
        gc.collect()

        self.device = device
        self.precision = precision
        self.max_tokens = llm_model_map[model_name]["max_tokens"]
        self.iree_module_dict = None
        self.compile()

    def compile(self) -> None:
        # this comes with keys: "vmfb", "config", and "temp_file_to_unlink".
        self.iree_module_dict = get_iree_compiled_module(
            self.tempfile_name, device=self.device, frontend="torch"
        )
        # TODO: delete the temp file

    def generate_images(
            self,
            prompt,
            ):
        history = []
        for iter in range(self.max_tokens):
            input_tensor = self.tokenizer(
                prompt, return_tensors="pt"
            ).input_ids
            device_inputs = [
                ireert.asdevicearray(
                    self.iree_module_dict["config"], input_tensor
                )
            ]
            if iter == 0:
                token = torch.tensor(
                    self.iree_module_dict["vmfb"]["run_initialize"](
                        *device_inputs
                    ).to_host()[0][0]
                )
            else:
                token = torch.tensor(
                    self.iree_module_dict["vmfb"]["run_forward"](
                        *device_inputs
                    ).to_host()[0][0]
                )

            history.append(token)
            yield self.tokenizer.decode(history)

            if token == llm_model_map["llama2_7b"]["stop_token"]:
                break

        for i in range(len(history)):
            if type(history[i]) != int:
                history[i] = int(history[i])
        result_output = self.tokenizer.decode(history)
        yield result_output


if __name__ == "__main__":
    lm = LanguageModel(
        "llama2_7b",
        hf_auth_token="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",
        device="cpu-task",
    )
    print("model loaded")
    for i in lm.chat("Hello, I am a robot."):
        print(i)
