from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
import tempfile
import torch_mlir


def prepare_sentence_tokens(hf_model: str, sentence: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    return torch.tensor([tokenizer.encode(sentence)])


class HfMaskedLM(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,  # The pretrained model name.
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
            torchscript=True,
        )
        self.model.eval()

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


hf_minilm_model = "gpt2"

test_input = prepare_sentence_tokens(
    hf_minilm_model, "this project is very interesting"
)

model = HfMaskedLM(hf_minilm_model)

print("Torch Golden Result: ", model(test_input))

fx_g = make_fx(
    model,
    decomposition_table=get_decompositions(
        [
            torch.ops.aten.embedding_dense_backward,
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.slice_backward,
            torch.ops.aten.select_backward,
            torch.ops.aten.norm.ScalarOpt_dim,
            torch.ops.aten.native_group_norm,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes,
        ]
    ),
)(test_input)

# print(fx_g.graph)

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
    (test_input),
    torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=True,
    verbose=False,
)

# module = torch_mlir.compile(
#     ts_g,
#     (test_input),
#     torch_mlir.OutputType.TOSA,
#     use_tracing=True,
#     verbose=False,
# )
# module.dump()


from shark.shark_inference import SharkInference

mlir_model = module
func_name = "forward"

shark_module = SharkInference(
    mlir_model, func_name, device="cpu", mlir_dialect="linalg"
)
shark_module.compile()


def shark_result(x):
    x_ny = x.detach().numpy()
    inputs = (x_ny,)
    result = shark_module.forward(inputs)
    return torch.from_numpy(result)


observed_out = shark_result(test_input)
print("SharkInference Result :", observed_out)
