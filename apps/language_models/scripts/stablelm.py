from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
import torch_mlir
from io import BytesIO
from shark.shark_inference import SharkInference

tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-base-alpha-7b")
inputs = tokenizer("What's your mood today?", return_tensors="pt")

class SLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-base-alpha-7b")

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]

slm_model = SLM()

fx_g = make_fx(
        slm_model,
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
    )(
            inputs['input_ids'], inputs['attention_mask']
    )

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
    ts_g, [inputs['input_ids'], inputs['attention_mask']], torch_mlir.OutputType.LINALG_ON_TENSORS, use_tracing=False, verbose=False
)

bytecode_stream = BytesIO()
mlir_module.operation.write_bytecode(bytecode_stream)
bytecode = bytecode_stream.getvalue()

shark_module = SharkInference(
    mlir_module=bytecode, device=args.device, mlir_dialect="tm_tensor"
)
shark_module.compile()

result = shark_module("forward", [inputs['input_ids'], inputs['attention_mask']])
