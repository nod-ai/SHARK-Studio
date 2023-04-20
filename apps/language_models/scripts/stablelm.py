import torch
import shark
from shark.shark_importer import import_with_fx
from shark.shark_inference import SharkInference
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch_mlir
from apps.stable_diffusion.src.utils import base_models, get_opt_flags
from apps.stable_diffusion.src.models.model_wrappers import replace_shape_str, get_vmfb_path_name
import os

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
# model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
# model.half().cuda()
# print(type(model))

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

inputs = tokenizer(prompt, return_tensors="pt")
# tokens = model.generate(
#   **inputs,
#   max_new_tokens=64,
#   temperature=0.7,
#   do_sample=True,
#   stopping_criteria=StoppingCriteriaList([StopOnTokens()])
# )
# print(tokenizer.decode(tokens[0], skip_special_tokens=True))


def get_input_info_for(model_info):
    dtype_config = {"f32": torch.float32, "i64": torch.int64}
    input_map = []
    for inp in model_info:
        shape = model_info[inp]["shape"]
        dtype = dtype_config[model_info[inp]["dtype"]]
        tensor = None
        if isinstance(shape, list):
            clean_shape = replace_shape_str(
                shape, 64, 512, 512, 1
            )
            if dtype == torch.int64:
                tensor = torch.randint(1, 3, tuple(clean_shape))
            else:
                tensor = torch.randn(*clean_shape).to(dtype)
        elif isinstance(shape, int):
            tensor = torch.tensor(shape).to(dtype)
        else:
            sys.exit("shape isn't specified correctly.")
        input_map.append(tensor)
    return input_map

def compile_module(shark_module, model_name, extra_args=[]):
    vmfb_path = get_vmfb_path_name(model_name)
    print(
        "No vmfb found. Compiling and saving to {}".format(
            vmfb_path
        )
    )
    path = shark_module.save_module(
        os.getcwd(), model_name, extra_args
    )
    shark_module.load_module(path, extra_args=extra_args)
    return shark_module

def get_slm():
    class SLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")

        def forward(self, input):
            return self.model(input)[0]

    slm_model = SLM()
    mlir_module, _ = import_with_fx(
        model=slm_model,
        inputs=get_input_info_for(base_models["clip"]),
        is_f16=False,
        debug=True,
        model_name="slm_model",
    )
    shark_module = SharkInference(
        mlir_module,
        # device=args.device,
        mlir_dialect="tm_tensor",
    )

    return compile_module(shark_module, "slm_model", get_opt_flags("clip", precision="fp32"))


slm = get_slm()
output = slm("forward", (inputs,))

# module = import_with_fx(model_wrapper, (inputs,))
# print(module)
# print(inputs["input_ids"])
# module = torch_mlir.compile(module, (inputs["input_ids"],), output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)

#print(module(function_name=fname, inputs= (inputs["input_ids"],)))


