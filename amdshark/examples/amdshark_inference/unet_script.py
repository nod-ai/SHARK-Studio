import torch
import numpy as np
from amdshark.amdshark_inference import AMDSharkInference
from amdshark.amdshark_importer import AMDSharkImporter


class UnetModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )
        self.model.eval()

    def forward(self, input):
        return self.model(input)


input = torch.randn(1, 3, 224, 224)

mlir_importer = AMDSharkImporter(
    UnetModule(),
    (input,),
    frontend="torch",
)

(vision_mlir, func_name), inputs, golden_out = mlir_importer.import_debug(
    tracing_required=False
)

amdshark_module = AMDSharkInference(vision_mlir, mlir_dialect="linalg")
amdshark_module.compile()
result = amdshark_module.forward((input,))
np.testing.assert_allclose(golden_out, result, rtol=1e-02, atol=1e-03)
