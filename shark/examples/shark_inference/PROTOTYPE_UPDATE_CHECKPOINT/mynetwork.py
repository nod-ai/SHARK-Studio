from shark.shark_inference import SharkInference
import torch
import torch.nn as nn
import torch.optim as optim
import torch_mlir
import torch.nn.functional as F
import os
os.environ["INITIAL_LINALG_IR"] = "linalgIR.mlir"
print("Picking initial Linalg IR from : "+str(os.environ["INITIAL_LINALG_IR"]))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x

net = Net()

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions

def compile_through_fx(model, inputs, outputType, fileToSaveTo, mlir_loc=None):
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
    )(inputs)

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

    print("Torchscript graph generated successfully")
    module = torch_mlir.compile(
        ts_g,
        inputs,
        outputType,
        use_tracing=False,
        verbose=False,
    )
    #return str(module)
    text_file = open(fileToSaveTo, "w")
    text_file.write(str(module))
    text_file.close()
    #print(str(module))

    mlir_model = str(module)
    func_name = "forward"
    shark_module = SharkInference(
        mlir_model, func_name, device="cpu", mlir_dialect="linalg"
    )
    shark_module.compile()

    return shark_module

def load_mlir(mlir_loc):
    import os

    if mlir_loc == None:
        return None
    with open(os.path.join(mlir_loc)) as f:
        mlir_module = f.read()
    return mlir_module

#print("INITIAL LINALG IR :-")
#print(load_mlir(os.environ["INITIAL_LINALG_IR"]))

print("WITH MODEL MARKING, INITIAL LINALG IR :-")
inputs = torch.rand(3, 3, 6, 7)
compile_through_fx(net, inputs, torch_mlir.OutputType.LINALG_ON_TENSORS, os.environ["INITIAL_LINALG_IR"])
print(load_mlir(os.environ["INITIAL_LINALG_IR"]))

net.load_state_dict(torch.load('model_checkpoint_1.pt')['model_state_dict'])
print("Loaded new checkpoints - 1")
inputs = torch.rand(3, 3, 6, 7)
compile_through_fx(net, inputs, torch_mlir.OutputType.RAW, "temp_linalg.mlir")

print("LINALG IR AFTER CHECKPOINT 1 :-")
print(load_mlir("temp_linalg.mlir"))

net.load_state_dict(torch.load('model_checkpoint_2.pt')['model_state_dict'])
print("Loaded new checkpoints - 2")
inputs = torch.rand(3, 3, 6, 7)
compile_through_fx(net, inputs, torch_mlir.OutputType.RAW, "temp_linalg.mlir")

print("LINALG IR AFTER CHECKPOINT 2 :-")
print(load_mlir("temp_linalg.mlir"))
