import torch
import numpy as np
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter


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

mlir_importer = SharkImporter(
    UnetModule(),
    (input,),
    frontend="torch",
)

(vision_mlir, func_name), inputs, golden_out = mlir_importer.import_debug(
    tracing_required=False
)

shark_module = SharkInference(vision_mlir, func_name, mlir_dialect="linalg")
shark_module.compile()
result = shark_module.forward((input,))
np.testing.assert_allclose(golden_out, result, rtol=1e-02, atol=1e-03)
