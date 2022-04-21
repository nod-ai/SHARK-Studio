import torch
import torch.nn as nn
import torchvision.models as models
from shark.shark_runner import SharkTrainer


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16, 2)
        self.train(True)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


input = torch.randn(10, 10)
labels = torch.randn(10, 2)

# shark_module = SharkInference(NeuralNet(), (input,), from_aot=True)
# results = shark_module.forward((input,))

shark_module = SharkTrainer(NeuralNet(), (input,), (labels,), from_aot=True)
results = shark_module.train((input,))

# print(results)

# input = torch.randn(1, 3, 224, 224)
# labels = torch.randn(1, 1000)

# class Resnet50Module(torch.nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.resnet = models.resnet50(pretrained=True)
        # self.train(True)

    # def forward(self, img):
        # return self.resnet.forward(img)

# shark_module = SharkTrainer(Resnet50Module(), (input,), (labels,), from_aot=True)
# results = shark_module.train((input,))
