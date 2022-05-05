import torch
import torch.nn as nn
from shark_runner import SharkInference, SharkTrainer


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16, 2)
        self.train(False)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


input = torch.randn(10, 10)
labels = torch.randn(1, 2)

shark_module = SharkInference(NeuralNet(), (input,))
results = shark_module.benchmark_forward((input,))

# TODO: Currently errors out in torch-mlir lowering pass.
# shark_trainer_module = SharkTrainer(
# NeuralNet(), (input,), (labels,), dynamic=True, from_aot=True
# )

# results = shark_trainer_module.train(input)

# print(results)
