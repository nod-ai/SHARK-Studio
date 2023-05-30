from shark.shark_inference import SharkInference
from apps.stable_diffusion.src.utils import (
    compile_through_fx,
    args,
)
import os
import torch

######################################
import torch_mlir
import torch._dynamo as torchdynamo
from shark.sharkdynamo.utils import make_shark_compiler
######################################

def get_test_model():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, padded_inp, slice_index):
            inp = padded_inp[:, :slice_index[0].item(), :]
            return torch.sum(inp)
    testModel = TestModel()
    print("Compiling testModel")
    shark_testModel, testModel_mlir = compile_through_fx(
        testModel,
        [torch.randint(3, (3,4,2), dtype=torch.float32), torch.randint(1,(1,))],
        extended_model_name="testModel",
        debug=False,
        generate_vmfb=True,
        save_dir=os.getcwd(),
        extra_args=[],
        base_model_id=None,
        model_name="testModel",
        precision=None,
        return_mlir=False,
    )
    print("Generated testModel.vmfb")
    return shark_testModel, testModel_mlir

import warnings, logging

warnings.simplefilter("ignore")


torchdynamo.reset()
torchdynamo.config.dynamic_shapes = True
torchdynamo.config.suppress_errors = True

@torchdynamo.optimize(
    make_shark_compiler(use_tracing=False, device="cuda", verbose=False)
)
def padded_to_nonpadded_sum(padded_inp, slice_index):
    inp = padded_inp[:, :slice_index[0].item(), :]
    return torch.sum(inp)

orig_inp = torch.randint(3, (3,4,2))
print("Original tensor :-")
print(orig_inp)
print("Shape: ", str(orig_inp.shape))
print("Original sum : ", torch.sum(orig_inp))

padded_inp = torch.nn.functional.pad(orig_inp, (0,0,0,5), "constant", 2)
print("Tensor after padding :-")
print(padded_inp)
print("Shape after padding : ", padded_inp.shape)
print("Sum after padding : ", torch.sum(padded_inp))

slice_tensor = torch.empty([1], dtype=torch.int64)
slice_tensor[0] = orig_inp.shape[1]
print("Output with SHARK : ", torch.from_numpy(padded_to_nonpadded_sum(padded_inp, slice_tensor)))
