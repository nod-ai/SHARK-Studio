import torch
import torch.fx
import torch_mlir
import torchvision.models as models

from shark.shark_inference import SharkInference
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions

class VisionModule(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, input):
        return self.model.forward(input)

timm_vit_model = models.vit_b_16(pretrained=True)

input = torch.randn(1, 3, 224, 224)

new_ts = VisionModule(timm_vit_model)

#fx_g = make_fx(model_inference)(input)
fx_g = make_fx(
        model_inference,
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
            input
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

module = torch_mlir.compile(ts_g, [input], torch_mlir.OutputType.LINALG_ON_TENSORS, use_tracing=True, verbose=False)

mlir_model = module
func_name = "forward"

shark_module = SharkInference(mlir_model, func_name, device="cpu", mlir_dialect="linalg")
shark_module.compile()

print(shark_module)