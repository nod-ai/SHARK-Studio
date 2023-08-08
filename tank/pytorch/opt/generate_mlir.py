import torch
import torch_mlir
from transformers import OPTConfig, GPT2Tokenizer

from hacked_hf_opt import OPTModel

configuration = OPTConfig()
configuration.return_dict = False

tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")


def make_mlir():
    model = OPTModel(configuration)
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    input_ids, attention_mask = inputs.data["input_ids"], inputs.data["attention_mask"]
    outputs = model(input_ids, attention_mask)

    module = torch_mlir.compile(
        model,
        (input_ids, attention_mask),
        output_type=torch_mlir.OutputType.TOSA,
        use_tracing=True,
    )

    asm_for_error_report = module.operation.get_asm(
        large_elements_limit=10, enable_debug_info=True)
    open("opt.torch.mlir", "w").write(asm_for_error_report)


def make_torchscript():
    model = OPTModel(configuration)
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    input_ids, attention_mask = inputs.data["input_ids"], inputs.data["attention_mask"]

    ts = torch.jit.freeze(torch.jit.trace(model, (input_ids, attention_mask)))
    print(ts.graph)


if __name__ == "__main__":
    make_mlir()
    # i have no idea why but because of freezing, if you try to TS first
    # then MLIR won't work
    make_torchscript()
