import torch
import torchvision.models as models
from shark_runner import SharkInference


class VisionModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train(False)

    def forward(self, input):
        return self.model.forward(input)


input = torch.randn(1, 3, 224, 224)

## The vision models present here: https://pytorch.org/vision/stable/models.html
vision_models_list = [
    models.resnet18(pretrained=True),
    models.alexnet(pretrained=True),
    models.vgg16(pretrained=True),
    models.squeezenet1_0(pretrained=True),
    models.densenet161(pretrained=True),
    models.inception_v3(pretrained=True),
    models.shufflenet_v2_x1_0(pretrained=True),
    models.mobilenet_v2(pretrained=True),
    models.mobilenet_v3_small(pretrained=True),
    models.resnext50_32x4d(pretrained=True),
    models.wide_resnet50_2(pretrained=True),
    models.mnasnet1_0(pretrained=True),
    models.efficientnet_b0(pretrained=True),
    models.regnet_y_400mf(pretrained=True),
    models.regnet_x_400mf(pretrained=True),
]

for i, vision_model in enumerate(vision_models_list):
    shark_module = SharkInference(
        VisionModule(vision_model),
        (input,),
    )
    shark_module.benchmark_forward((input,))
