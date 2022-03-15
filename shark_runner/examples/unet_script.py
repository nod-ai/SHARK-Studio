import torch
import numpy as np
import os
import sys
from shark_runner import SharkInference

# Currently not supported aten.transpose_conv2d missing.
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
        self.train(False)

    def forward(self, input):
        return self.model(input)


input = torch.randn(1, 3, 224, 224)

print(input)
shark_module = SharkInference(
    UnetModule(),
    (input,),
)
shark_module.forward((input,))
print(input)
