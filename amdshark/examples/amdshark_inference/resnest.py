import torch
import torchvision.models as models
from amdshark.amdshark_inference import AMDSharkInference
from amdshark.amdshark_importer import AMDSharkImporter

torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)


class ResnestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "zhanghang1989/ResNeSt", "resnest50", pretrained=True
        )
        self.model.eval()

    def forward(self, input):
        return self.model.forward(input)


input = torch.randn(1, 3, 224, 224)


mlir_importer = AMDSharkImporter(
    ResnestModule(),
    (input,),
    frontend="torch",
)

(vision_mlir, func_name), inputs, golden_out = mlir_importer.import_debug(
    tracing_required=True
)

print(golden_out)

amdshark_module = AMDSharkInference(vision_mlir, mlir_dialect="linalg")
amdshark_module.compile()
result = amdshark_module.forward((input,))
print("Obtained result", result)
