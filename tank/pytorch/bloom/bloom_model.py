### Please do `pip install transformers==4.21.2` before running this script.

### To run the complete bloom model: pass as argument "--config bloom".

import argparse
import torch
import torch_mlir
from transformers import BloomTokenizerFast, BloomForSequenceClassification

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from shark.shark_inference import SharkInference

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
p.add_argument(
    "--prompt",
    type=str,
    default="Hello, my dog is cute",
    help="the text prompt to use",
)
p.add_argument("--device", type=str, default="cpu", help="the device to use")
p.add_argument("--seed", type=int, default=0, help="the random seed")
p.add_argument(
    "--config",
    type=str,
    default="bloom-560m",
    help="the configuration of model to use",
)
args = p.parse_args()

torch.manual_seed(args.seed)

model_config = "bigscience/" + args.config
tokenizer = BloomTokenizerFast.from_pretrained(model_config)
test_input = tokenizer(args.prompt, return_tensors="pt")["input_ids"]


class HuggingFaceLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BloomForSequenceClassification.from_pretrained(
            model_config
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


model = HuggingFaceLanguage()
actual_out = model(test_input)

# import numpy as np
# test_input_ny = test_input.detach().numpy()
# input_tuple = (test_input_ny,)
# np.savez('inputs.npz', *input_tuple)
# output_ny = actual_out.detach().numpy()
# output_tuple = (output_ny,)
# np.savez('golden_out.npz', *output_tuple)

fx_g = make_fx(
    model,
    decomposition_table=get_decompositions(
        [
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes,
        ]
    ),
)(test_input)

# # print(fx_g.graph)

fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
fx_g.recompile()


def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


strip_overloads(fx_g)

ts_g = torch.jit.script(fx_g)

module = torch_mlir.compile(
    ts_g,
    [test_input],
    torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=True,
    verbose=False,
)
# # module.dump()

mlir_model = module
func_name = "forward"

shark_module = SharkInference(
    mlir_model, func_name, device=args.device, mlir_dialect="tm_tensor"
)
shark_module.compile()


def shark_result(x):
    x_ny = x.detach().numpy()
    inputs = (x_ny,)
    result = shark_module.forward(inputs)
    return torch.from_numpy(result)


observed_out = shark_result(test_input)

print("Golden result:", actual_out)
print("SHARK result:", observed_out)
