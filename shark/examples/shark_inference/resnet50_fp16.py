from shark.shark_inference import SharkInference
from shark.parser import shark_args

import torch
import numpy as np
import sys
import torchvision.models as models
import torch_mlir

torch.manual_seed(0)


class VisionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.train(False)

    def forward(self, input):
        return self.model.forward(input)


model = VisionModule()
test_input = torch.randn(1, 3, 224, 224)
actual_out = model(test_input)

test_input_fp16 = test_input.to(device=torch.device("cuda"), dtype=torch.half)
model_fp16 = model.half()
model_fp16.eval()
model_fp16.to("cuda")
actual_out_fp16 = model_fp16(test_input_fp16)

ts_g = torch.jit.trace(model_fp16, [test_input_fp16])

module = torch_mlir.compile(
    ts_g,
    (test_input_fp16),
    torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=True,
    verbose=False,
)

# from contextlib import redirect_stdout

# with open('resnet50_fp16_linalg_ir.mlir', 'w') as f:
#     with redirect_stdout(f):
#         print(module.operation.get_asm())

mlir_model = module
func_name = "forward"

shark_module = SharkInference(mlir_model, device="cuda", mlir_dialect="linalg")
shark_module.compile()


def shark_result(x):
    x_ny = x.cpu().detach().numpy()
    inputs = (x_ny,)
    result = shark_module.forward(inputs)
    return torch.from_numpy(result)


observed_out = shark_result(test_input_fp16)

print("Golden result:", actual_out_fp16)
print("SHARK result:", observed_out)

actual_out_fp16 = actual_out_fp16.to(device=torch.device("cpu"))

print(
    torch.testing.assert_allclose(
        actual_out_fp16, observed_out, rtol=1e-2, atol=1e-2
    )
)
